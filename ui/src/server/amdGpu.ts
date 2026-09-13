import { ChildProcess, execFile, spawn } from 'child_process';
import { promisify } from 'util';
import os from 'os';
import { GPUApiResponse, GpuInfo } from '@/types';
import { amdGpuStatsScriptPath } from '@/server/amdGpuScript';

const execFileAsync = promisify(execFile);

/**
 * AMD ROCm GPU sampling, shared by the /api/gpu route and the system monitor.
 *
 * Linux hosts keep using `amd-smi` exactly like the AMD fork did. Windows ROCm
 * ships no SMI tool at all, so there the numbers come from a small Python
 * helper (see amdGpuScript.ts) that reads HIP, the PDH GPU engine counters and
 * D3DKMT. Windows exposes no temperature/power/fan telemetry for Radeon cards
 * without the vendor ADL/ADLX APIs, so those fields stay 0 rather than being
 * invented.
 *
 * Device indices are the HIP device order, which is the order PyTorch uses for
 * `--gpu-ids`, so the GPU selected in the UI is the GPU training actually uses.
 */

/** One JSON line from the Windows helper. */
interface HelperGpu {
  index?: unknown;
  name?: string;
  driverVersion?: string;
  pciBus?: number | null;
  totalMiB?: unknown;
  freeMiB?: unknown;
  utilizationGpu?: unknown;
}

interface HelperSample {
  gpus?: HelperGpu[];
  error?: string | null;
}

interface PythonCommand {
  command: string;
  args: string[];
}

/** How long to wait for the helper's first sample before giving up on it. */
const HELPER_START_TIMEOUT_MS = 20_000;
/** amd-smi has no loop mode, so the Linux fallback polls slower than the tick. */
const AMD_SMI_POLL_MS = 2000;
/** Largest stdout buffer kept while waiting for a newline (a line is ~1 KB). */
const MAX_STDOUT_BUFFER = 1 << 20;

/**
 * Reads a number that amd-smi may report as a number, a numeric string or the
 * string "N/A". parseFloat never throws, so a try/catch around it would let
 * NaN through - this is the check that actually keeps NaN out of the UI.
 */
function finiteNumber(value: unknown, fallback = 0): number {
  if (typeof value === 'number') return Number.isFinite(value) ? value : fallback;
  const parsed = parseFloat(String(value));
  return Number.isFinite(parsed) ? parsed : fallback;
}

function amdResponse(gpus: GpuInfo[], error?: string): GPUApiResponse {
  return {
    hasNvidiaSmi: false,
    isMac: false,
    isAMD: true,
    gpus,
    error,
  };
}

// ---------------------------------------------------------------------------
// Windows: Python helper
// ---------------------------------------------------------------------------
function helperGpuToGpuInfo(gpu: HelperGpu): GpuInfo {
  const total = Math.max(Math.round(finiteNumber(gpu.totalMiB)), 0);
  const free = Math.min(Math.max(Math.round(finiteNumber(gpu.freeMiB)), 0), total);
  const used = Math.max(total - free, 0);
  return {
    index: Math.round(finiteNumber(gpu.index)),
    name: gpu.name || 'AMD GPU',
    driverVersion: gpu.driverVersion || 'ROCm',
    // Not exposed by Windows ROCm; 0 renders as "0" instead of a fake reading.
    temperature: 0,
    utilization: {
      gpu: Math.round(finiteNumber(gpu.utilizationGpu)),
      memory: total > 0 ? Math.round((used / total) * 100) : 0,
    },
    memory: { total, free, used },
    power: { draw: 0, limit: 0 },
    clocks: { graphics: 0, memory: 0 },
    fan: { speed: 0 },
  };
}

function parseHelperSample(stdout: string): HelperSample | null {
  const lines = stdout
    .split(/\r?\n/)
    .map(line => line.trim())
    .filter(line => line.length > 0);
  // Watch mode may have emitted several lines; the last one is the newest.
  for (let index = lines.length - 1; index >= 0; index--) {
    try {
      const parsed = JSON.parse(lines[index]) as HelperSample;
      if (parsed && Array.isArray(parsed.gpus)) return parsed;
    } catch {
      // Not a sample line (stray output); keep looking.
    }
  }
  return null;
}

function helperSampleToResponse(sample: HelperSample): GPUApiResponse {
  const gpus = (sample.gpus ?? []).map(helperGpuToGpuInfo).sort((a, b) => a.index - b.index);
  const error = sample.error ?? (gpus.length === 0 ? 'The HIP runtime reported no AMD ROCm devices' : undefined);
  return amdResponse(gpus, error ?? undefined);
}

let pythonCommand: PythonCommand | null | undefined;

/** First interpreter that exists and can import ctypes (the helper needs it). */
async function findPython(): Promise<PythonCommand | null> {
  if (pythonCommand !== undefined) return pythonCommand;
  const candidates: PythonCommand[] = [];
  if (process.env.AI_TOOLKIT_PYTHON) {
    candidates.push({ command: process.env.AI_TOOLKIT_PYTHON, args: [] });
  }
  candidates.push({ command: 'python', args: [] });
  candidates.push({ command: 'python3', args: [] });
  if (os.platform() === 'win32') {
    candidates.push({ command: 'py', args: ['-3'] });
  }
  for (const candidate of candidates) {
    try {
      await execFileAsync(candidate.command, [...candidate.args, '-c', 'import ctypes'], {
        timeout: 15_000,
        windowsHide: true,
      });
      pythonCommand = candidate;
      return candidate;
    } catch {
      // Try the next interpreter.
    }
  }
  pythonCommand = null;
  return null;
}

async function sampleAmdWindows(): Promise<GPUApiResponse | null> {
  const python = await findPython();
  if (!python) return null;
  let stdout: string;
  try {
    const result = await execFileAsync(
      python.command,
      [...python.args, amdGpuStatsScriptPath(), '--once'],
      { timeout: 60_000, maxBuffer: 8 * 1024 * 1024, windowsHide: true },
    );
    stdout = result.stdout;
  } catch (error) {
    // The helper reports its diagnostics on stderr; a bare "Command failed"
    // message would hide the actual cause.
    const failure = error as { stderr?: string; message?: string };
    const detail = (failure.stderr ?? '').trim() || failure.message || 'unknown error';
    console.error('AMD GPU helper failed:', detail);
    return amdResponse([], `AMD GPU helper failed: ${detail}`);
  }
  const sample = parseHelperSample(stdout);
  if (!sample) {
    return amdResponse([], 'The AMD GPU helper produced no parsable sample');
  }
  return helperSampleToResponse(sample);
}

// ---------------------------------------------------------------------------
// Linux: amd-smi
// ---------------------------------------------------------------------------
interface AmdSmiValue {
  value?: unknown;
}

interface AmdSmiStaticGpu {
  gpu?: unknown;
  asic?: { market_name?: string };
  driver?: { version?: string };
  limit?: { max_power?: AmdSmiValue; ppt0?: { max_power_limit?: AmdSmiValue } };
}

interface AmdSmiMetricGpu {
  usage?: { gfx_activity?: AmdSmiValue } | null;
  temperature?: { hotspot?: AmdSmiValue };
  mem_usage?: { total_vram?: AmdSmiValue; used_vram?: AmdSmiValue; free_visible_vram?: AmdSmiValue };
  power?: { socket_power?: AmdSmiValue };
  clock?: { gfx_0?: { clk?: AmdSmiValue }; mem_0?: { clk?: AmdSmiValue } };
  fan?: { usage?: AmdSmiValue };
}

function amdSmiPowerLimit(device: AmdSmiStaticGpu): number {
  const direct = device.limit?.max_power?.value;
  if (direct !== undefined) return finiteNumber(direct);
  return finiteNumber(device.limit?.ppt0?.max_power_limit?.value);
}

async function amdSmiJson(args: string[]): Promise<unknown> {
  const { stdout } = await execFileAsync('amd-smi', args, {
    timeout: 30_000,
    maxBuffer: 32 * 1024 * 1024,
  });
  return JSON.parse(stdout);
}

async function sampleAmdLinux(): Promise<GPUApiResponse | null> {
  try {
    await execFileAsync('amd-smi', ['version'], { timeout: 15_000 });
  } catch {
    return null;
  }

  let staticData: unknown;
  let metricData: unknown;
  try {
    [staticData, metricData] = await Promise.all([
      amdSmiJson(['static', '--json']),
      amdSmiJson(['metric', '--json']),
    ]);
  } catch (error) {
    const message = error instanceof Error ? error.message : String(error);
    console.error('amd-smi failed:', message);
    return amdResponse([], `amd-smi failed: ${message}`);
  }

  const staticGpus = (staticData as { gpu_data?: AmdSmiStaticGpu[] })?.gpu_data ?? [];
  const metricGpus = (metricData as { gpu_data?: AmdSmiMetricGpu[] })?.gpu_data ?? [];

  const gpus: GpuInfo[] = [];
  staticGpus.forEach((device, position) => {
    const index = Math.trunc(finiteNumber(device.gpu, position));
    const metrics = metricGpus[index];
    // Integrated GPUs have no usage block at all, and their metrics are the
    // "N/A" strings that used to leak NaN into the UI. Skip them.
    if (!metrics || typeof metrics.usage !== 'object' || metrics.usage === null) return;
    const total = finiteNumber(metrics.mem_usage?.total_vram?.value);
    const used = finiteNumber(metrics.mem_usage?.used_vram?.value);
    const free = finiteNumber(metrics.mem_usage?.free_visible_vram?.value, Math.max(total - used, 0));
    gpus.push({
      index,
      name: device.asic?.market_name ?? `AMD GPU ${index}`,
      driverVersion: device.driver?.version ?? 'ROCm',
      temperature: Math.trunc(finiteNumber(metrics.temperature?.hotspot?.value)),
      utilization: {
        gpu: Math.trunc(finiteNumber(metrics.usage?.gfx_activity?.value)),
        memory: total > 0 ? Math.round((used / total) * 100) : 0,
      },
      memory: { total, free, used },
      power: {
        draw: finiteNumber(metrics.power?.socket_power?.value),
        limit: amdSmiPowerLimit(device),
      },
      clocks: {
        graphics: Math.trunc(finiteNumber(metrics.clock?.gfx_0?.clk?.value)),
        memory: Math.trunc(finiteNumber(metrics.clock?.mem_0?.clk?.value)),
      },
      fan: { speed: finiteNumber(metrics.fan?.usage?.value) },
    });
  });

  const error = gpus.length === 0 ? 'amd-smi reported no AMD ROCm devices' : undefined;
  return amdResponse(gpus, error);
}

// ---------------------------------------------------------------------------
// Public surface
// ---------------------------------------------------------------------------

/**
 * One-shot AMD sample: `null` when no AMD backend exists on this machine
 * (so callers can fall back to reporting "no GPU support"), otherwise a
 * response that may still carry an `error` for the UI to display.
 */
export async function sampleAmdGpuOnce(): Promise<GPUApiResponse | null> {
  if (os.platform() === 'win32') return sampleAmdWindows();
  return sampleAmdLinux();
}

/**
 * Keeps AMD stats flowing at monitor cadence.
 *
 * `nvidia-smi` can be kept resident with `-lms`, but there is no AMD
 * equivalent, and MONITOR_TICK_MS is 500 ms - far too fast to spawn a helper
 * per tick. So on Windows one Python helper is started in `--watch` mode and
 * its JSON lines are consumed as they arrive; on Linux the (slower) amd-smi
 * poll runs on its own interval and the monitor reuses the latest value.
 */
export class AmdGpuWatcher {
  private child: ChildProcess | null = null;
  private stdoutBuffer = '';
  private stderrTail = '';
  private pollTimer: NodeJS.Timeout | null = null;
  private restartTimer: NodeJS.Timeout | null = null;
  private polling = false;
  private starting = false;
  private stopped = false;
  private sample: GPUApiResponse | null = null;

  constructor(
    private readonly onSample: (response: GPUApiResponse) => void,
    private readonly intervalMs: number,
  ) {}

  get latestSample(): GPUApiResponse | null {
    return this.sample;
  }

  /**
   * Resolves true when an AMD backend answered (even with zero devices, so the
   * UI can show AMD's own error instead of an nvidia-smi one).
   */
  async start(): Promise<boolean> {
    this.stopped = false;
    if (os.platform() === 'win32') return this.startHelperStream();
    return this.startPolling();
  }

  stop(): void {
    this.stopped = true;
    if (this.pollTimer) {
      clearInterval(this.pollTimer);
      this.pollTimer = null;
    }
    if (this.restartTimer) {
      clearTimeout(this.restartTimer);
      this.restartTimer = null;
    }
    if (this.child) {
      try {
        this.child.kill();
      } catch {
        // already gone
      }
      this.child = null;
    }
  }

  private async startPolling(): Promise<boolean> {
    const first = await sampleAmdGpuOnce();
    if (!first || this.stopped) {
      return false;
    }
    this.publish(first);
    // amd-smi spawns are not free, so poll slower than the 500 ms tick and let
    // the monitor reuse the most recent value in between.
    const interval = Math.max(this.intervalMs, AMD_SMI_POLL_MS);
    this.pollTimer = setInterval(() => void this.poll(), interval);
    return true;
  }

  private async poll(): Promise<void> {
    if (this.stopped || this.polling) return;
    this.polling = true;
    try {
      const response = await sampleAmdGpuOnce();
      if (response && !this.stopped) this.publish(response);
    } catch (error) {
      console.error('AMD GPU: amd-smi sampling failed:', error);
    } finally {
      this.polling = false;
    }
  }

  private async startHelperStream(): Promise<boolean> {
    if (this.starting) return this.sample !== null;
    this.starting = true;
    try {
      const python = await findPython();
      if (!python) {
        console.warn('AMD GPU: no Python interpreter with ctypes found; cannot read Windows ROCm GPUs');
        return false;
      }
      let child: ChildProcess;
      try {
        child = spawn(
          python.command,
          [...python.args, amdGpuStatsScriptPath(), '--watch', String(Math.max(this.intervalMs, 100))],
          { stdio: ['ignore', 'pipe', 'pipe'], windowsHide: true },
        );
      } catch (error) {
        console.error('AMD GPU: could not start the statistics helper:', error);
        return false;
      }

      this.child = child;
      this.stdoutBuffer = '';
      this.stderrTail = '';

      child.stdout?.on('data', (chunk: Buffer) => this.onHelperData(chunk.toString()));
      child.stderr?.on('data', (chunk: Buffer) => {
        // Keep the tail only; it is surfaced if the helper dies.
        this.stderrTail = (this.stderrTail + chunk.toString()).slice(-2000);
      });
      child.on('error', (error: NodeJS.ErrnoException) => {
        console.error('AMD GPU helper error:', error.message);
        if (this.child === child) this.child = null;
      });
      child.on('exit', (code: number | null) => {
        if (this.child !== child) return;
        this.child = null;
        if (this.stopped) return;
        if (this.stderrTail.trim()) console.warn('AMD GPU helper output:', this.stderrTail.trim());
        // Restart only once we have seen a sample; a helper that never worked
        // would otherwise be respawned forever.
        if (this.sample && !this.restartTimer) {
          this.restartTimer = setTimeout(() => {
            this.restartTimer = null;
            void this.startHelperStream();
          }, 5000);
        } else if (!this.sample) {
          console.warn(`AMD GPU helper exited before its first sample (code ${code})`);
        }
      });

      // Wait for the first sample so the caller learns whether AMD is usable.
      const deadline = Date.now() + HELPER_START_TIMEOUT_MS;
      while (!this.stopped && this.sample === null && this.child !== null && Date.now() < deadline) {
        await delay(100);
      }
      if (this.sample === null) {
        console.warn(
          'AMD GPU helper produced no sample' + (this.stderrTail.trim() ? `: ${this.stderrTail.trim()}` : ''),
        );
        this.stop();
        return false;
      }
      return true;
    } finally {
      this.starting = false;
    }
  }

  private onHelperData(text: string): void {
    this.stdoutBuffer += text;
    if (this.stdoutBuffer.length > MAX_STDOUT_BUFFER) {
      this.stdoutBuffer = this.stdoutBuffer.slice(-MAX_STDOUT_BUFFER);
    }
    const lines = this.stdoutBuffer.split(/\r?\n/);
    this.stdoutBuffer = lines.pop() ?? '';
    for (const line of lines) {
      const trimmed = line.trim();
      if (!trimmed) continue;
      let parsed: HelperSample;
      try {
        parsed = JSON.parse(trimmed) as HelperSample;
      } catch {
        continue;
      }
      if (!parsed || !Array.isArray(parsed.gpus)) continue;
      this.publish(helperSampleToResponse(parsed));
    }
  }

  private publish(response: GPUApiResponse): void {
    this.sample = response;
    this.onSample(response);
  }
}

function delay(ms: number): Promise<void> {
  return new Promise(resolve => setTimeout(resolve, ms));
}
