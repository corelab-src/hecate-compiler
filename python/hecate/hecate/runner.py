import ctypes
import weakref
import re
import inspect
from subprocess import Popen
from collections.abc import Iterable

# import torch

import os
import time
import numpy as np
import numpy.ctypeslib as npcl
from pathlib import Path

import json

hecate_dir = Path(os.environ["HECATE"])
hecateBuild = hecate_dir / "build"

if not hecateBuild.is_dir():  # We expect that this is library path
    hecateBuild = hecate_dir

libpath = hecateBuild / "lib"
os.environ["PATH"] = str(libpath) + os.pathsep + os.environ["PATH"]

# Library Wrapper (lw)
lw = None


def set_lw_functions(lw_handle):
    # Init VM functions
    lw_handle.initFullVM.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_bool,
    ]
    lw_handle.initFullVM.restype = ctypes.c_void_p
    lw_handle.initClientVM.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_bool,
    ]
    lw_handle.initClientVM.restype = ctypes.c_void_p
    lw_handle.initServerVM.argtypes = [
        ctypes.c_char_p,
        ctypes.c_int64,
        ctypes.c_int64,
        ctypes.c_bool,
    ]
    lw_handle.initServerVM.restype = ctypes.c_void_p

    # Init Contexts
    lw_handle.create_context.argtypes = [ctypes.c_char_p]
    lw_handle.load.argtypes = [ctypes.c_void_p, ctypes.c_char_p, ctypes.c_char_p]
    lw_handle.loadClient.argtypes = [ctypes.c_void_p, ctypes.c_void_p]
    lw_handle.getArgLen.argtypes = [ctypes.c_void_p]
    lw_handle.getArgLen.restype = ctypes.c_int64
    lw_handle.getResLen.argtypes = [ctypes.c_void_p]
    lw_handle.getResLen.restype = ctypes.c_int64

    # Optional (only for some libraries)
    if hasattr(lw_handle, "setEpoch"):
        lw_handle.setEpoch.argtypes = [ctypes.c_void_p, ctypes.c_int64, ctypes.c_int]

    # Encrypt/Decrypt
    lw_handle.encrypt.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
        ctypes.c_int,
    ]
    lw_handle.decrypt.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
    ]
    lw_handle.decrypt_result.argtypes = [
        ctypes.c_void_p,
        ctypes.c_int64,
        ctypes.POINTER(ctypes.c_double),
    ]

    # Ciphertext access
    lw_handle.getResIdx.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    lw_handle.getResIdx.restype = ctypes.c_int64
    lw_handle.getCtxt.argtypes = [ctypes.c_void_p, ctypes.c_int64]
    lw_handle.getCtxt.restype = ctypes.c_void_p

    # Runner
    lw_handle.preprocess.argtypes = [ctypes.c_void_p]
    lw_handle.run.argtypes = [ctypes.c_void_p]

    # Debug
    lw_handle.setDebug.argtypes = [ctypes.c_void_p, ctypes.c_bool]

    # GPU (optional)
    if hasattr(lw_handle, "setToGPU"):
        lw_handle.setToGPU.argtypes = [ctypes.c_void_p, ctypes.c_bool]


def init_lw():
    global lw

    libso_name = "lib" + run_library + "_HEVM.so"
    if not os.path.exists(libpath / libso_name):
        raise FileNotFoundError(f"Library {libso_name} not found in {libpath}")

    lw = ctypes.CDLL(libpath / libso_name)
    os.environ["PATH"] = str(libpath) + os.pathsep + os.environ["PATH"]
    set_lw_functions(lw)


run_library = "SEAL"
run_hardware = "CPU"
config_N = 2 << 17
config_L = 17


def setLibnHW(argv=None):
    global run_library
    global run_hardware
    global config_N
    global config_L
    LibnHW_mapping = {
        "HEAAN": ["GPU", "CPU"],
        "SEAL": ["CPU"],
        "HEONGPU": ["GPU"],
        # "TOY" : ["CPU", "GPU"],
    }
    HWnLib_mapping = {}
    for support_libs, support_HWs in LibnHW_mapping.items():
        for HW in support_HWs:
            if HW in HWnLib_mapping:
                HWnLib_mapping[HW].append(support_libs)
            else:
                HWnLib_mapping[HW] = [support_libs]
    # TODO : How about using getopt
    if len(argv) >= 4:
        if argv[3].upper() in LibnHW_mapping.keys():
            run_library = argv[3].upper()
            if len(argv) == 5:
                if argv[4].upper() in LibnHW_mapping[run_library]:
                    run_hardware = argv[4].upper()
                else:
                    print("Not supported", argv[4].upper())
                    print("Suppoerted hardware :", LibnHW_mapping[run_library])
                    exit()
            else:
                run_hardware = LibnHW_mapping[run_library][0]
        elif argv[3].upper() in HWnLib_mapping.keys():
            run_hardware = argv[3].upper()
            if len(argv) == 5:
                if argv[4].upper() in HWnLib_mapping[run_hardware]:
                    run_library = argv[4].upper()
                else:
                    print("Not supported", argv[4].upper())
                    print("Suppoerted library :", HWnLib_mapping[run_hardware])
                    exit()
            else:
                run_library = HWnLib_mapping[run_hardware][0]
        else:
            print("Not supported", argv[3])
            print("Require Bootstrapping-supported Library")
            print("Supported library :", LibnHW_mapping.keys())
            print("Supported hardware :", HWnLib_mapping.keys())
            exit()
    else:
        # For default
        run_library = list(LibnHW_mapping.keys())[0]
        run_hardware = LibnHW_mapping[run_library][0]
    config_name = f"profiled_{run_library}_{run_hardware}.json"
    with open(str(hecate_dir) + "/" + config_name, "r") as f:
        run_config = json.load(f)
        config_N = run_config["polynomialDegree"]
        config_L = run_config["levelUpperBound"]


class HEVM:
    def __init__(
        self, path=str((Path.home() / ".hevm" / "default").absolute()), option="full"
    ):
        global run_library
        global run_hardware
        global config_N
        global config_L
        init_lw()

        self.option = option
        self.boot_cnt = 0
        if path == str((Path.home() / ".hevm" / "default").absolute()):
            # When user does not set path
            path = str((Path.home() / ".hevm" / run_library).absolute())
        if not Path(path).is_dir():
            print(
                f"Press Any key to generate {run_library} key files (or just kill with ctrl+c)"
            )
            input()
            Path(path).mkdir(parents=True)
            lw.create_context(path.encode("utf-8"))

        if option == "full":
            if run_hardware == "GPU":
                self.vm = lw.initFullVM(path.encode("utf-8"), config_N, config_L, True)
            elif run_hardware == "CPU":
                self.vm = lw.initFullVM(path.encode("utf-8"), config_N, config_L, False)
        elif option == "client":
            self.vm = lw.initClientVM(path.encode("utf-8"), config_N, config_L)
        elif option == "server":
            self.vm = lw.initServerVM(path.encode("utf-8"), config_N, config_L)

    # def load (self, func,   preprocess=True, const_path =str( (Path(func_dir) / "_hecate_{func}.cst").absoluate() ), hevm_path = str(Path(func_dir) / "_hecate_{func}.hevm"), func_dir = str(Path.cwd()), ) :
    def load(self, const_path, hevm_path, preprocess=True):
        if not Path(const_path).is_file():
            raise Exception(f"No file exists in const_path {const_path}")
        if not Path(hevm_path).is_file():
            raise Exception(f"No file exists in hevm_path {hevm_path}")
        if self.option == "full" or self.option == "server":
            lw.load(self.vm, const_path.encode("utf-8"), hevm_path.encode("utf-8"))
        elif self.option == "client":
            lw.loadClient(self.vm, const_path.encode("utf-8"))
        if preprocess:
            lw.preprocess(self.vm)
        else:
            raise Exception("Not implemented in SEAL_HEVM")

        self.arglen = lw.getArgLen(self.vm)
        self.reslen = lw.getResLen(self.vm)
        self.hevm_path = hevm_path
        self.const_path = const_path

    def run(self):
        lw.run(self.vm)

    def setInput(self, i, data):
        if not isinstance(data, np.ndarray):
            data = np.array(data, dtype=np.float64)
        carr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lw.encrypt(self.vm, i, carr, len(data))

    def setEpoch(self, i, data):
        # if not isinstance(data, np.ndarray) :
        #     data = np.array(data, dtype=np.float64)
        # carr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
        lw.setEpoch(self.vm, i, data)

    def setDebug(self, enable):
        lw.setDebug(self.vm, enable)

    def setToGPU(self, ongpu):
        lw.setToGPU(self.vm, ongpu)

    def getOutput(self):
        slot_size = config_N >> 1
        result = np.zeros((self.reslen, slot_size), dtype=np.float64)
        data = np.zeros(slot_size, dtype=np.float64)

        for i in range(self.reslen):
            carr = data.ctypes.data_as(ctypes.POINTER(ctypes.c_double))
            lw.decrypt_result(self.vm, i, carr)
            result[i] = data
        return result

    def printer(self, latency, rms, loop_count=1, mem_usage=0.0):
        import re

        bench = re.search(r"optimized/(.*)/(.*)\.(.*)\._", self.hevm_path)
        print("==================================================")
        print("--------------- Runtime Option ------------------")
        print("compiler:", bench.group(1))
        print("benchname:", bench.group(2))
        print("waterline:", bench.group(3))
        print("library:", run_library)
        print("device:", run_hardware)
        print("loop count:", loop_count)
        print("----------------- Test Results -------------------")
        print("code size (bytes):", os.path.getsize(self.hevm_path))
        print("const size (bytes):", os.path.getsize(self.const_path))
        print("latency (s):", latency)
        print("rms:", rms)
        print("==================================================")
        print()
