"""Lists the Vulkan physical devices a process sees (ctypes, no vulkan-tools needed).

Used to check GPU pinning for the WebGPU runs: with VK_LOADER_DEVICE_ID_FILTER=0x2805 the Vulkan loader (system
1.4.357 and the one bundled with Chrome for Testing both read it) should expose only the RTX 4060 Ti (GPU 1).

Usage: [VK_LOADER_DEVICE_ID_FILTER=0x2805] python vk_devices.py [path to libvulkan.so.1]
"""

from __future__ import annotations

import ctypes
import json
import sys

VK_STRUCTURE_TYPE_APPLICATION_INFO = 0
VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO = 1


class AppInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("pApplicationName", ctypes.c_char_p),
        ("applicationVersion", ctypes.c_uint32),
        ("pEngineName", ctypes.c_char_p),
        ("engineVersion", ctypes.c_uint32),
        ("apiVersion", ctypes.c_uint32),
    ]


class InstanceInfo(ctypes.Structure):
    _fields_ = [
        ("sType", ctypes.c_int),
        ("pNext", ctypes.c_void_p),
        ("flags", ctypes.c_uint32),
        ("pApplicationInfo", ctypes.POINTER(AppInfo)),
        ("enabledLayerCount", ctypes.c_uint32),
        ("ppEnabledLayerNames", ctypes.c_void_p),
        ("enabledExtensionCount", ctypes.c_uint32),
        ("ppEnabledExtensionNames", ctypes.c_void_p),
    ]


class Properties(ctypes.Structure):
    # VkPhysicalDeviceProperties up to deviceName; the rest (pipelineCacheUUID, limits, sparse) as padding.
    _fields_ = [
        ("apiVersion", ctypes.c_uint32),
        ("driverVersion", ctypes.c_uint32),
        ("vendorID", ctypes.c_uint32),
        ("deviceID", ctypes.c_uint32),
        ("deviceType", ctypes.c_int),
        ("deviceName", ctypes.c_char * 256),
        ("rest", ctypes.c_ubyte * 1024),
    ]


def main() -> None:
    lib = ctypes.CDLL(sys.argv[1] if len(sys.argv) > 1 else "libvulkan.so.1")
    app = AppInfo(VK_STRUCTURE_TYPE_APPLICATION_INFO, None, b"vk_devices", 1, b"none", 1, (1 << 22) | (1 << 12))
    info = InstanceInfo(VK_STRUCTURE_TYPE_INSTANCE_CREATE_INFO, None, 0, ctypes.pointer(app), 0, None, 0, None)
    instance = ctypes.c_void_p()
    result = lib.vkCreateInstance(ctypes.byref(info), None, ctypes.byref(instance))
    if result != 0:
        raise SystemExit(f"vkCreateInstance failed: {result}")
    count = ctypes.c_uint32(0)
    lib.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), None)
    devices = (ctypes.c_void_p * count.value)()
    lib.vkEnumeratePhysicalDevices(instance, ctypes.byref(count), devices)
    out = []
    for device in devices:
        props = Properties()
        lib.vkGetPhysicalDeviceProperties(ctypes.c_void_p(device), ctypes.byref(props))
        out.append(
            {"name": props.deviceName.decode(), "vendor_id": hex(props.vendorID), "device_id": hex(props.deviceID)}
        )
    lib.vkDestroyInstance(instance, None)
    print(json.dumps(out))


if __name__ == "__main__":
    main()
