import fs from 'fs';
import os from 'os';
import path from 'path';

/**
 * Windows ROCm ships neither `amd-smi` nor `rocm-smi`, and no AMD telemetry
 * library is installed with it either, so the Linux AMD data path simply
 * cannot work there. The helper below reads the same kind of information out
 * of the Windows APIs that do exist:
 *
 *   HIP runtime (amdhip64*.dll) - device count, name, total/free VRAM, PCI bus
 *   PDH performance counters    - per-adapter GPU engine utilization
 *   D3DKMT                      - maps a counter instance's LUID to a PCI bus
 *   registry                    - display driver version
 *
 * HIP device order is the order PyTorch uses for `--gpu-ids`, so the `index`
 * reported here is the same index training jobs select.
 *
 * It is embedded as a string (rather than shipped as a .py file) so it works
 * no matter how the Next.js server is bundled or where it is started from.
 */
export const AMD_GPU_STATS_SCRIPT = String.raw`"""AMD GPU statistics for Windows ROCm hosts.

ROCm for Windows ships no amd-smi/rocm-smi, so the UI's Linux data path cannot
work there. This helper collects the same information from the Windows APIs
that do exist.

Usage:
    python amdGpuStats.py --once
    python amdGpuStats.py --watch 500

One JSON object is written to stdout per sample:

    {"gpus": [{"index": 0, "name": "AMD Radeon AI PRO R9700", "pciBus": 13,
               "driverVersion": "32.0.31021.1015", "totalMiB": 32624.0,
               "freeMiB": 32472.9, "usedMiB": 151.1, "utilizationGpu": 1.7}],
     "error": null}

Diagnostics go to stderr so stdout stays a clean JSON stream.
"""

import ctypes
import ctypes.wintypes as wt
import glob
import json
import os
import sys
import time

PDH_FMT_DOUBLE = 0x00000200
PDH_CSTATUS_VALID_DATA = 0x00000000
GPU_ENGINE_COUNTER = r"\GPU Engine(*)\Utilization Percentage"
GPU_ADAPTER_MEMORY_COUNTER = r"\GPU Adapter Memory(*)\Dedicated Usage"
KMTQAITYPE_ADAPTERADDRESS = 6
VIDEO_CLASS_KEY = r"SYSTEM\CurrentControlSet\Control\Class\{4d36e968-e325-11ce-bfc1-08002be10318}"

BYTES_PER_MIB = 1048576.0


def log(message):
    sys.stderr.write("amdGpuStats: " + str(message) + "\n")
    sys.stderr.flush()


# ---------------------------------------------------------------------------
# HIP runtime: device identity, live VRAM, PCI bus
# ---------------------------------------------------------------------------
def rocm_bin_dirs():
    """Directories that may hold amdhip64*.dll, most specific first."""
    candidates = []
    for variable in ("ROCM_PATH", "ROCM_HOME", "HIP_PATH"):
        value = os.environ.get(variable)
        if value:
            candidates.append(os.path.join(value, "bin"))
            candidates.append(value)
    candidates.extend(sorted(glob.glob(r"C:\Program Files\AMD\ROCm\*\bin"), reverse=True))
    for entry in os.environ.get("PATH", "").split(os.pathsep):
        if entry and "rocm" in entry.lower():
            candidates.append(entry)
    unique = []
    seen = set()
    for directory in candidates:
        key = directory.lower()
        if directory and os.path.isdir(directory) and key not in seen:
            seen.add(key)
            unique.append(directory)
    return unique


_hip = None


def load_hip():
    global _hip
    if _hip is not None:
        return _hip
    problems = []
    for directory in rocm_bin_dirs():
        for dll in sorted(glob.glob(os.path.join(directory, "amdhip64*.dll")), reverse=True):
            try:
                _hip = ctypes.WinDLL(dll)
                return _hip
            except OSError as exc:
                problems.append(dll + ": " + str(exc))
    for dll in ("amdhip64.dll", "amdhip64_7.dll"):
        try:
            _hip = ctypes.WinDLL(dll)
            return _hip
        except OSError as exc:
            problems.append(dll + ": " + str(exc))
    detail = "; ".join(problems) if problems else "no amdhip64 DLL found"
    raise OSError("could not load the HIP runtime (" + detail + ")")


def pci_bus_from_id(pci_id):
    """hipDeviceGetPCIBusId returns e.g. "0000:0d:00.0" -> bus 13 (hex)."""
    parts = (pci_id or "").split(":")
    if len(parts) < 2:
        return None
    try:
        return int(parts[1], 16)
    except ValueError:
        return None


def read_hip_devices():
    hip = load_hip()
    count = ctypes.c_int(0)
    if hip.hipGetDeviceCount(ctypes.byref(count)) != 0:
        raise OSError("hipGetDeviceCount failed")
    devices = []
    for index in range(count.value):
        name = ctypes.create_string_buffer(256)
        if hip.hipDeviceGetName(name, 256, index) != 0:
            log("hipDeviceGetName failed for device " + str(index))
            continue
        free_bytes = ctypes.c_size_t(0)
        total_bytes = ctypes.c_size_t(0)
        has_memory = hip.hipMemGetInfo(ctypes.byref(free_bytes), ctypes.byref(total_bytes)) == 0
        bus_id = ctypes.create_string_buffer(64)
        has_bus = hip.hipDeviceGetPCIBusId(bus_id, 64, index) == 0
        devices.append({
            "index": index,
            "name": name.value.decode("utf-8", "replace"),
            "pciBus": pci_bus_from_id(bus_id.value.decode("ascii", "replace")) if has_bus else None,
            "totalMiB": round(total_bytes.value / BYTES_PER_MIB, 1) if has_memory else 0.0,
            "freeMiB": round(free_bytes.value / BYTES_PER_MIB, 1) if has_memory else 0.0,
        })
        if not has_memory:
            log("hipMemGetInfo failed for device " + str(index))
    return devices


# ---------------------------------------------------------------------------
# PDH: per-adapter GPU engine utilization
# ---------------------------------------------------------------------------
class _PdhCounterValue(ctypes.Structure):
    _fields_ = [("CStatus", wt.DWORD), ("doubleValue", ctypes.c_double)]


class _PdhCounterValueItem(ctypes.Structure):
    _fields_ = [("szName", wt.LPWSTR), ("FmtValue", _PdhCounterValue)]


def luid_and_engine(instance):
    """Split "..._luid_0x00000000_0x00014E15_phys_0_eng_0_engtype_3D".

    Engine instances are prefixed with "pid_<n>_"; adapter memory instances
    start straight at "luid_", so search for the marker without the leading
    underscore.
    """
    start = instance.find("luid_")
    if start < 0:
        return None, None
    parts = instance[start + len("luid_"):].split("_")
    if len(parts) < 2:
        return None, None
    engine = None
    marker = instance.find("engtype_")
    if marker >= 0:
        engine = instance[marker + len("engtype_"):]
    return (parts[0], parts[1]), engine


class EngineCounters(object):
    """GPU engine utilization per adapter LUID, from the Windows PDH counters."""

    def __init__(self):
        self.pdh = ctypes.WinDLL("pdh")
        self.query = ctypes.c_void_p()
        self.counter = ctypes.c_void_p()
        self.memory_counter = None
        self.primed = False
        if self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.query)) != 0:
            raise OSError("PdhOpenQuery failed")
        # The English counter names resolve on localized Windows too; the
        # localized PdhAddCounter would not.
        if self.pdh.PdhAddEnglishCounterW(self.query, GPU_ENGINE_COUNTER, 0, ctypes.byref(self.counter)) != 0:
            raise OSError("the GPU Engine counters are unavailable")
        memory_counter = ctypes.c_void_p()
        if self.pdh.PdhAddEnglishCounterW(
                self.query, GPU_ADAPTER_MEMORY_COUNTER, 0, ctypes.byref(memory_counter)) == 0:
            self.memory_counter = memory_counter
        else:
            log("the GPU Adapter Memory counters are unavailable")

    def _formatted_array(self, counter):
        """[(instance name, value)] for one counter, dropping invalid samples."""
        size = wt.DWORD(0)
        count = wt.DWORD(0)
        self.pdh.PdhGetFormattedCounterArrayW(
            counter, PDH_FMT_DOUBLE, ctypes.byref(size), ctypes.byref(count), None)
        if size.value == 0 or count.value == 0:
            return []
        buffer = ctypes.create_string_buffer(size.value)
        if self.pdh.PdhGetFormattedCounterArrayW(
                counter, PDH_FMT_DOUBLE, ctypes.byref(size), ctypes.byref(count), buffer) != 0:
            return []
        items = ctypes.cast(buffer, ctypes.POINTER(_PdhCounterValueItem))
        rows = []
        for position in range(count.value):
            item = items[position]
            if item.FmtValue.CStatus == PDH_CSTATUS_VALID_DATA:
                rows.append((item.szName or "", item.FmtValue.doubleValue))
        return rows

    def collect(self):
        """{luid: percent}, the busiest engine of each adapter."""
        if not self.primed:
            # A percentage counter has no meaning until two samples exist.
            self.pdh.PdhCollectQueryData(self.query)
            time.sleep(0.15)
            self.primed = True
        if self.pdh.PdhCollectQueryData(self.query) != 0:
            return {}
        per_engine = {}
        for name, value in self._formatted_array(self.counter):
            if value <= 0:
                continue
            luid, engine = luid_and_engine(name)
            if luid is None:
                continue
            key = (luid, engine or "unknown")
            per_engine[key] = per_engine.get(key, 0.0) + value
        per_luid = {}
        for (luid, _engine), value in per_engine.items():
            # Values are per process, so they add up across processes but more
            # than one engine can legitimately be busy at once: report the
            # busiest engine, the way Task Manager does.
            busiest = min(value, 100.0)
            if busiest > per_luid.get(luid, 0.0):
                per_luid[luid] = busiest
        return per_luid

    def collect_memory(self):
        """{luid: dedicated VRAM in MiB} per adapter.

        hipMemGetInfo only accounts for the calling process, so it reports a
        training job running in another process as ~0. This driver-side counter
        is the one that sees the whole device. Call after collect() so both
        counters are read from the same query sample.
        """
        if self.memory_counter is None:
            return {}
        per_luid = {}
        for name, value in self._formatted_array(self.memory_counter):
            if value <= 0:
                continue
            luid, _engine = luid_and_engine(name)
            if luid is None:
                continue
            per_luid[luid] = per_luid.get(luid, 0.0) + value / BYTES_PER_MIB
        return per_luid


# ---------------------------------------------------------------------------
# D3DKMT: adapter LUID -> PCI bus number
# ---------------------------------------------------------------------------
class _Luid(ctypes.Structure):
    _fields_ = [("LowPart", wt.DWORD), ("HighPart", wt.LONG)]


class _OpenAdapterFromLuid(ctypes.Structure):
    _fields_ = [("AdapterLuid", _Luid), ("hAdapter", wt.UINT)]


class _QueryAdapterInfo(ctypes.Structure):
    _fields_ = [
        ("hAdapter", wt.UINT),
        ("Type", wt.UINT),
        ("pPrivateDriverData", ctypes.c_void_p),
        ("PrivateDriverDataSize", wt.UINT),
    ]


class _CloseAdapter(ctypes.Structure):
    _fields_ = [("hAdapter", wt.UINT)]


class _AdapterAddress(ctypes.Structure):
    _fields_ = [("BusNumber", wt.UINT), ("DeviceNumber", wt.UINT), ("FunctionNumber", wt.UINT)]


_gdi32 = None


def luid_to_bus(high, low):
    """PCI bus of an adapter LUID; None for software/virtual adapters."""
    global _gdi32
    if _gdi32 is None:
        _gdi32 = ctypes.WinDLL("gdi32")
        _gdi32.D3DKMTOpenAdapterFromLuid.restype = ctypes.c_long
        _gdi32.D3DKMTQueryAdapterInfo.restype = ctypes.c_long
        _gdi32.D3DKMTCloseAdapter.restype = ctypes.c_long
    opened = _OpenAdapterFromLuid(_Luid(low & 0xFFFFFFFF, high), 0)
    if _gdi32.D3DKMTOpenAdapterFromLuid(ctypes.byref(opened)) != 0:
        return None
    try:
        address = _AdapterAddress()
        query = _QueryAdapterInfo(
            opened.hAdapter,
            KMTQAITYPE_ADAPTERADDRESS,
            ctypes.cast(ctypes.byref(address), ctypes.c_void_p),
            ctypes.sizeof(address),
        )
        if _gdi32.D3DKMTQueryAdapterInfo(ctypes.byref(query)) != 0:
            return None
        # The iGPU and virtual display adapters report an all-ones address.
        if address.BusNumber >= 0xFFFF:
            return None
        return address.BusNumber
    finally:
        _gdi32.D3DKMTCloseAdapter(ctypes.byref(_CloseAdapter(opened.hAdapter)))


# ---------------------------------------------------------------------------
# Registry: display driver version
# ---------------------------------------------------------------------------
def read_driver_versions():
    """{adapter description: driver version} for installed display adapters."""
    versions = {}
    try:
        import winreg
    except ImportError:
        return versions
    try:
        root = winreg.OpenKey(winreg.HKEY_LOCAL_MACHINE, VIDEO_CLASS_KEY)
    except OSError as exc:
        log("display class registry key unavailable: " + str(exc))
        return versions
    with root:
        position = 0
        while True:
            try:
                subkey_name = winreg.EnumKey(root, position)
            except OSError:
                break
            position += 1
            try:
                with winreg.OpenKey(root, subkey_name) as subkey:
                    description = winreg.QueryValueEx(subkey, "DriverDesc")[0]
                    version = winreg.QueryValueEx(subkey, "DriverVersion")[0]
            except OSError:
                continue
            if description and version and description not in versions:
                versions[description] = str(version)
    return versions


def driver_version_for(name, versions):
    if not versions:
        return ""
    if name in versions:
        return versions[name]
    lowered = (name or "").lower()
    for description, version in versions.items():
        other = description.lower()
        if lowered and (lowered in other or other in lowered):
            return version
    for description, version in versions.items():
        if "radeon" in description.lower():
            return version
    return ""


# ---------------------------------------------------------------------------
# Sampling
# ---------------------------------------------------------------------------
def bus_for(high, low):
    try:
        return luid_to_bus(int(high, 16), int(low, 16))
    except ValueError:
        return None


def build_sample(counters, versions):
    devices = read_hip_devices()
    utilization = {}
    dedicated = {}
    if counters is not None:
        try:
            for (high, low), percent in counters.collect().items():
                bus = bus_for(high, low)
                if bus is None:
                    continue
                if percent > utilization.get(bus, 0.0):
                    utilization[bus] = percent
        except Exception as exc:
            # Utilization is a nice-to-have; memory stats still matter.
            log("gpu engine counters failed: " + str(exc))
        try:
            for (high, low), mib in counters.collect_memory().items():
                bus = bus_for(high, low)
                if bus is None:
                    continue
                dedicated[bus] = dedicated.get(bus, 0.0) + mib
        except Exception as exc:
            log("gpu adapter memory counters failed: " + str(exc))

    gpus = []
    for device in devices:
        total = device["totalMiB"]
        bus = device["pciBus"]
        hip_free = device["freeMiB"]
        hip_used = total - hip_free if total > 0 and hip_free <= total else 0.0
        # Prefer the device-wide counter over HIP, which only sees this
        # process: with a training job in another process HIP reports ~0.
        used = dedicated.get(bus, hip_used) if bus is not None else hip_used
        used = max(min(used, total), 0.0)
        free = max(total - used, 0.0)
        gpus.append({
            "index": device["index"],
            "name": device["name"],
            "driverVersion": driver_version_for(device["name"], versions),
            "pciBus": bus,
            "totalMiB": total,
            "freeMiB": round(free, 1),
            "usedMiB": round(used, 1),
            "utilizationGpu": round(utilization.get(bus, 0.0), 1) if bus is not None else 0.0,
        })
    return {"gpus": gpus, "error": None}


def main(argv):
    watch_ms = None
    if "--watch" in argv:
        position = argv.index("--watch")
        try:
            watch_ms = int(argv[position + 1])
        except (IndexError, ValueError):
            watch_ms = 500

    counters = None
    try:
        counters = EngineCounters()
    except Exception as exc:
        log("gpu engine counters unavailable: " + str(exc))

    versions = read_driver_versions()

    while True:
        try:
            sample = build_sample(counters, versions)
        except Exception as exc:
            sample = {"gpus": [], "error": str(exc)}
        sys.stdout.write(json.dumps(sample) + "\n")
        sys.stdout.flush()
        if watch_ms is None:
            return 0
        time.sleep(max(watch_ms, 100) / 1000.0)


if __name__ == "__main__":
    sys.exit(main(sys.argv))
`;

let scriptPath: string | null = null;

/**
 * Materialize the helper under the OS temp directory and return its path. The
 * file is rewritten only when the source changes, so the path stays stable and
 * can be run by hand while debugging.
 */
export function amdGpuStatsScriptPath(): string {
  if (scriptPath) return scriptPath;
  const directory = path.join(os.tmpdir(), 'ai-toolkit-amd-gpu');
  const target = path.join(directory, 'amdGpuStats.py');
  let existing: string | null = null;
  try {
    existing = fs.readFileSync(target, 'utf8');
  } catch {
    existing = null;
  }
  if (existing !== AMD_GPU_STATS_SCRIPT) {
    fs.mkdirSync(directory, { recursive: true });
    fs.writeFileSync(target, AMD_GPU_STATS_SCRIPT, 'utf8');
  }
  scriptPath = target;
  return target;
}
