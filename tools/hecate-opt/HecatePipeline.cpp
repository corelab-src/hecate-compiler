#include "HecateOptMain.h"

using namespace llvm;

using namespace mlir;
using namespace hecate;

void registerHecatePipeline(cl::opt<std::string> &outputFilename) {

  static HecateOptOptions opt;

  PassPipelineRegistration<>(
      "eva", "Perform waterline rescaling and early modswitch",
      [&](OpPassManager &pm) {
        std::string dir;
        std::string stem;
        if (outputFilename != "-") {
          std::filesystem::path outputName(outputFilename.getValue());
          stem = outputName.stem();
          dir = outputName.parent_path();
        }
        if (opt.enable_check_smu)
          pm.addPass(hecate::earth::createSMUChecker());

        pm.addNestedPass<func::FuncOp>(hecate::earth::createWaterlineRescaling(
            {opt.waterline, opt.output_val}));
        pm.addNestedPass<func::FuncOp>(hecate::earth::createEarlyModswitch());

        if (opt.enable_check_smu)
          pm.addPass(hecate::earth::createSMUChecker());

        pm.addPass(createCSEPass());
        pm.addPass(createCanonicalizerPass());

        if (opt.enable_printer)
          pm.addPass(createLocationSnapshotPass(
              OpPrintingFlags().enableDebugInfo(false, false),
              dir + "/" + stem + ".earth.mlir", "earth"));

        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createEarthToCKKSConversionPass());

        pm.addNestedPass<func::FuncOp>(
            hecate::ckks::createUpscaleToMulcpConversionPass());

        if (opt.enable_printer)
          pm.addPass(createLocationSnapshotPass(
              OpPrintingFlags().enableDebugInfo(false, false),
              dir + "/" + stem + ".ckks.mlir", "ckks"));
        pm.addNestedPass<func::FuncOp>(hecate::ckks::createRemoveLevel());
        pm.addNestedPass<func::FuncOp>(hecate::ckks::createReuseBuffer());
        pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
        pm.addNestedPass<func::FuncOp>(
            hecate::ckks::createEmitHEVM({dir + "/" + stem}));
      });

  PassPipelineRegistration<>(
      "snr", "Perform SNR rescaling and early modswitch",
      [&](OpPassManager &pm) {
        std::string dir;
        std::string stem;
        if (outputFilename != "-") {
          std::filesystem::path outputName(outputFilename.getValue());
          stem = outputName.stem();
          dir = outputName.parent_path();
        }

        if (opt.enable_check_smu)
          pm.addPass(hecate::earth::createSMUChecker());

        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createSNRRescaling({opt.waterline, opt.output_val}));
        pm.addNestedPass<func::FuncOp>(hecate::earth::createEarlyModswitch());

        if (opt.enable_check_smu)
          pm.addPass(hecate::earth::createSMUChecker());

        pm.addPass(createCSEPass());
        pm.addPass(createCanonicalizerPass());

        if (opt.enable_printer)
          pm.addPass(createLocationSnapshotPass(
              OpPrintingFlags().enableDebugInfo(false, false),
              dir + "/" + stem + ".earth.mlir", "earth"));

        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createEarthToCKKSConversionPass());

        pm.addNestedPass<func::FuncOp>(
            hecate::ckks::createUpscaleToMulcpConversionPass());

        if (opt.enable_printer)
          pm.addPass(createLocationSnapshotPass(
              OpPrintingFlags().enableDebugInfo(false, false),
              dir + "/" + stem + ".ckks.mlir", "ckks"));
        pm.addNestedPass<func::FuncOp>(hecate::ckks::createRemoveLevel());
        pm.addNestedPass<func::FuncOp>(hecate::ckks::createReuseBuffer());
        pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
        pm.addNestedPass<func::FuncOp>(
            hecate::ckks::createEmitHEVM({dir + "/" + stem}));
      });

  PassPipelineRegistration<>(
      "elasm", "Perform ELASM exploration ", [&](OpPassManager &pm) {
        std::string dir;
        std::string stem;
        if (outputFilename != "-") {
          std::filesystem::path outputName(outputFilename.getValue());
          stem = outputName.stem();
          dir = outputName.parent_path();
        }

        if (opt.enable_check_smu)
          pm.addPass(hecate::earth::createSMUChecker());

        pm.addNestedPass<func::FuncOp>(hecate::earth::createELASMExplorer(
            {opt.waterline, opt.output_val, opt.parallel_elasm,
             opt.num_iter_elasm, opt.beta_elasm, opt.gamma_elasm}));
        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createScaleManagementScheduler());
        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createSNRRescaling({opt.waterline, opt.output_val}));
        pm.addNestedPass<func::FuncOp>(hecate::earth::createUpscaleBubbling());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addNestedPass<func::FuncOp>(hecate::earth::createEarlyModswitch());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());
        pm.addNestedPass<func::FuncOp>(hecate::earth::createErrorEstimator());
        pm.addNestedPass<func::FuncOp>(hecate::earth::createLatencyEstimator());

        if (opt.enable_check_smu)
          pm.addPass(hecate::earth::createSMUChecker());

        if (opt.enable_printer)
          pm.addPass(createLocationSnapshotPass(
              OpPrintingFlags().enableDebugInfo(false, false),
              dir + "/" + stem + ".earth.mlir", "earth"));

        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createEarthToCKKSConversionPass());

        if (opt.enable_printer)
          pm.addPass(createLocationSnapshotPass(
              OpPrintingFlags().enableDebugInfo(false, false),
              dir + "/" + stem + ".ckks.mlir", "ckks"));

        pm.addNestedPass<func::FuncOp>(
            hecate::ckks::createUpscaleToMulcpConversionPass());
        pm.addNestedPass<func::FuncOp>(hecate::ckks::createRemoveLevel());
        pm.addNestedPass<func::FuncOp>(hecate::ckks::createReuseBuffer());
        pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
        pm.addNestedPass<func::FuncOp>(
            hecate::ckks::createEmitHEVM({dir + "/" + stem}));
      });

  PassPipelineRegistration<>(
      "dacapo", "Perform automatic bootstrapping placement",
      [&](OpPassManager &pm) {
        std::string dir;
        std::string stem;
        if (outputFilename != "-") {
          std::filesystem::path outputName(outputFilename.getValue());
          stem = outputName.stem();
          dir = outputName.parent_path();
        }
        if (opt.enable_check_smu)
          pm.addPass(hecate::earth::createSMUChecker());

        pm.addNestedPass<func::FuncOp>(hecate::earth::createRemoveBootstrap());
        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createBypassDetection({opt.waterline, 0.5}));
        pm.addNestedPass<func::FuncOp>(hecate::earth::createCandidateSelection(
            {opt.waterline, opt.output_val}));
        pm.addNestedPass<func::FuncOp>(hecate::earth::createDaCapoPlanner(
            {opt.waterline, opt.output_val}));
        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createBootstrapPlacement());
        pm.addNestedPass<func::FuncOp>(hecate::earth::createProactiveRescaling(
            {opt.waterline, opt.output_val}));
        pm.addNestedPass<func::FuncOp>(hecate::earth::createEarlyModswitch());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addPass(mlir::createCSEPass());

        if (opt.enable_check_smu)
          pm.addPass(hecate::earth::createSMUChecker());

        pm.addPass(createCSEPass());
        pm.addPass(createCanonicalizerPass());

        if (opt.enable_printer)
          pm.addPass(createLocationSnapshotPass(
              OpPrintingFlags().enableDebugInfo(false, false),
              dir + "/" + stem + ".earth.mlir", "earth"));

        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createEarthToCKKSConversionPass());

        pm.addNestedPass<func::FuncOp>(
            hecate::ckks::createUpscaleToMulcpConversionPass());

        if (opt.enable_printer)
          pm.addPass(createLocationSnapshotPass(
              OpPrintingFlags().enableDebugInfo(false, false),
              dir + "/" + stem + ".ckks.mlir", "ckks"));
        pm.addNestedPass<func::FuncOp>(hecate::ckks::createRemoveLevel());
        pm.addNestedPass<func::FuncOp>(hecate::ckks::createReuseBuffer());
        pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
        pm.addNestedPass<func::FuncOp>(
            hecate::ckks::createEmitHEVM({dir + "/" + stem}));
      });
  PassPipelineRegistration<>(
      "pars", "Perform Scale Management on bootstrapped models",
      [&](OpPassManager &pm) {
        std::string dir;
        std::string stem;
        if (outputFilename != "-") {
          std::filesystem::path outputName(outputFilename.getValue());
          stem = outputName.stem();
          dir = outputName.parent_path();
        }
        if (opt.enable_check_smu)
          pm.addPass(hecate::earth::createSMUChecker());

        pm.addNestedPass<func::FuncOp>(hecate::earth::createProactiveRescaling(
            {opt.waterline, opt.output_val}));
        pm.addNestedPass<func::FuncOp>(hecate::earth::createEarlyModswitch());

        if (opt.enable_check_smu)
          pm.addPass(hecate::earth::createSMUChecker());

        pm.addPass(createCSEPass());
        pm.addPass(createCanonicalizerPass());

        if (opt.enable_printer)
          pm.addPass(createLocationSnapshotPass(
              OpPrintingFlags().enableDebugInfo(false, false),
              dir + "/" + stem + ".earth.mlir", "earth"));

        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createEarthToCKKSConversionPass());

        pm.addNestedPass<func::FuncOp>(
            hecate::ckks::createUpscaleToMulcpConversionPass());

        if (opt.enable_printer)
          pm.addPass(createLocationSnapshotPass(
              OpPrintingFlags().enableDebugInfo(false, false),
              dir + "/" + stem + ".ckks.mlir", "ckks"));
        pm.addNestedPass<func::FuncOp>(hecate::ckks::createRemoveLevel());
        pm.addNestedPass<func::FuncOp>(hecate::ckks::createReuseBuffer());
        pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
        pm.addNestedPass<func::FuncOp>(
            hecate::ckks::createEmitHEVM({dir + "/" + stem}));
      });

  PassPipelineRegistration<>(
      "halo", "Perform Loop Optimization on HE programs",
      [&](OpPassManager &pm) {
        std::string dir;
        std::string stem;
        if (outputFilename != "-") {
          std::filesystem::path outputName(outputFilename.getValue());
          stem = outputName.stem();
          dir = outputName.parent_path();
        }
        if (opt.enable_check_smu)
          pm.addPass(hecate::earth::createSMUChecker());

        pm.addNestedPass<func::FuncOp>(hecate::earth::createRemoveBootstrap());
        pm.addNestedPass<func::FuncOp>(earth::createPackLoopVariables());
        pm.addNestedPass<func::FuncOp>(hecate::earth::createApplyDaCapoToLoop(
            {opt.waterline, opt.output_val, true}));
        pm.addNestedPass<func::FuncOp>(hecate::earth::createLoopUnroll());
        pm.addPass(mlir::createSymbolDCEPass());

        pm.addNestedPass<func::FuncOp>(hecate::earth::createProactiveRescaling(
            {opt.waterline, opt.output_val}));
        pm.addNestedPass<func::FuncOp>(hecate::earth::createEarlyModswitch());
        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createFlexibleBootstrap());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createCanonicalizerPass());
        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createEarthToEarthConversionPass());
        pm.addNestedPass<func::FuncOp>(
            earth::createElideConstant({dir + "/../../traced/cst/"}));

        if (opt.enable_check_smu)
          pm.addPass(hecate::earth::createSMUChecker());

        if (opt.enable_printer)
          pm.addPass(createLocationSnapshotPass(
              OpPrintingFlags().enableDebugInfo(false, false),
              dir + "/" + stem + ".earth.mlir", "earth"));

        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createEarthToCKKSConversionPass());

        pm.addNestedPass<func::FuncOp>(
            hecate::ckks::createUpscaleToMulcpConversionPass());
        pm.addNestedPass<func::FuncOp>(
            hecate::ckks::createSCFToDpsSCFConversionPass());

        if (opt.enable_printer)
          pm.addPass(createLocationSnapshotPass(
              OpPrintingFlags().enableDebugInfo(false, false),
              dir + "/" + stem + ".ckks.mlir", "ckks"));
        pm.addNestedPass<func::FuncOp>(hecate::ckks::createRemoveLevel());
        pm.addNestedPass<func::FuncOp>(hecate::ckks::createReuseBuffer());
        pm.addNestedPass<func::FuncOp>(hecate::ckks::createAllocLoopBuffer());
        pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
        pm.addNestedPass<func::FuncOp>(
            hecate::ckks::createEmitHEVM({dir + "/" + stem}));
      });
}
