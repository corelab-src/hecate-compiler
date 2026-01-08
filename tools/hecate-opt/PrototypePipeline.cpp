#include "HecateOptMain.h"

using namespace llvm;

using namespace mlir;
using namespace hecate;

void registerPrototypePipeline(cl::opt<std::string> &outputFilename) {

  static HecateOptOptions opt;
  PassPipelineRegistration<>(
      "dacapo_flex", "Perform automatic bootstrapping placement",
      [&](OpPassManager &pm) {
        std::string dir;
        std::string stem;
        if (outputFilename != "-") {
          std::filesystem::path outputName(outputFilename.getValue());
          stem = outputName.stem();
          dir = outputName.parent_path();
          if (!std::filesystem::exists(dir)) {
            std::filesystem::create_directories(dir);
          }
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
        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createFlexibleBootstrap());
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
      "simple_loop", "Perform Loop Optimization on HE programs",
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
        pm.addNestedPass<func::FuncOp>(hecate::earth::createApplyDaCapoToLoop(
            {opt.waterline, opt.output_val}));

        pm.addNestedPass<func::FuncOp>(hecate::earth::createProactiveRescaling(
            {opt.waterline, opt.output_val}));
        pm.addNestedPass<func::FuncOp>(hecate::earth::createEarlyModswitch());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createCanonicalizerPass());

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
  PassPipelineRegistration<>(
      "packed_loop", "Perform Loop Optimization on HE programs",
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
            {opt.waterline, opt.output_val}));

        pm.addNestedPass<func::FuncOp>(hecate::earth::createProactiveRescaling(
            {opt.waterline, opt.output_val}));
        pm.addNestedPass<func::FuncOp>(hecate::earth::createEarlyModswitch());
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
  PassPipelineRegistration<>(
      "unroll_loop", "Perform Loop Optimization on HE programs",
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
        /* pm.addNestedPass<func::FuncOp>(earth::createPackLoopVariables()); */
        pm.addNestedPass<func::FuncOp>(hecate::earth::createApplyDaCapoToLoop(
            {opt.waterline, opt.output_val, true}));
        pm.addNestedPass<func::FuncOp>(hecate::earth::createLoopUnroll());
        pm.addPass(mlir::createSymbolDCEPass());

        pm.addNestedPass<func::FuncOp>(hecate::earth::createProactiveRescaling(
            {opt.waterline, opt.output_val}));
        pm.addNestedPass<func::FuncOp>(hecate::earth::createEarlyModswitch());
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

  PassPipelineRegistration<>(
      "unroll_factor_loop", "Perform Loop Optimization on HE programs",
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
            hecate::earth::createLoopUnroll({opt.unroll_factor}));
        pm.addNestedPass<func::FuncOp>(earth::createPackLoopVariables());
        pm.addNestedPass<func::FuncOp>(hecate::earth::createApplyDaCapoToLoop(
            {opt.waterline, opt.output_val}));
        pm.addPass(mlir::createSymbolDCEPass());

        pm.addNestedPass<func::FuncOp>(hecate::earth::createProactiveRescaling(
            {opt.waterline, opt.output_val}));
        pm.addNestedPass<func::FuncOp>(hecate::earth::createEarlyModswitch());
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

  PassPipelineRegistration<>(
      "packed_unroll_loop", "Perform Loop Optimization on HE programs",
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

  PassPipelineRegistration<>(
      "packed_flex_loop", "Perform Loop Optimization on HE programs",
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
            {opt.waterline, opt.output_val}));

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

  PassPipelineRegistration<>(
      "flex_loop", "Perform Loop Optimization on HE programs",
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
        pm.addNestedPass<func::FuncOp>(hecate::earth::createApplyDaCapoToLoop(
            {opt.waterline, opt.output_val}));

        pm.addNestedPass<func::FuncOp>(hecate::earth::createProactiveRescaling(
            {opt.waterline, opt.output_val}));
        pm.addNestedPass<func::FuncOp>(hecate::earth::createEarlyModswitch());
        pm.addNestedPass<func::FuncOp>(
            hecate::earth::createFlexibleBootstrap());
        pm.addPass(mlir::createCSEPass());
        pm.addPass(mlir::createCanonicalizerPass());

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
  PassPipelineRegistration<>(
      "unroll_flex_loop", "Perform Loop Optimization on HE programs",
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
        /* pm.addNestedPass<func::FuncOp>(earth::createPackLoopVariables()); */
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
            earth::createElideConstant({dir + "/../../traced" + "/"}));

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

  PassPipelineRegistration<>(
      "level-preserved-shape-independent",
      "Perform Trivial Function Support on FHE programs",
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
        pm.addPass(hecate::earth::createCallerBootstrapMgmt());
        // pm.addNestedPass<func::FuncOp>(
        //     hecate::earth::createProactiveRescaling({opt.waterline,
        //     opt.output_val}));
        // pm.addNestedPass<func::FuncOp>(hecate::earth::createEarlyModswitch());
        // pm.addNestedPass<func::FuncOp>(
        //     hecate::earth::createFlexibleBootstrap());
        //   pm.addPass(mlir::createCSEPass());
        //   pm.addPass(mlir::createCanonicalizerPass());
        //   pm.addNestedPass<func::FuncOp>(
        //       hecate::earth::createEarthToEarthConversionPass());
        //   pm.addNestedPass<func::FuncOp>(
        //       earth::createElideConstant({dir + "/../../traced" + "/"}));
        //
        //   if (opt.enable_check_smu)
        //     pm.addPass(hecate::earth::createSMUChecker());
        //
        //   if (opt.enable_printer)
        //     pm.addPass(createLocationSnapshotPass(
        //         OpPrintingFlags().enableDebugInfo(false, false),
        //         dir + "/" + stem + ".earth.mlir", "earth"));
        //
        //   pm.addNestedPass<func::FuncOp>(
        //       hecate::earth::createEarthToCKKSConversionPass());
        //
        //   pm.addNestedPass<func::FuncOp>(
        //       hecate::ckks::createUpscaleToMulcpConversionPass());
        //   pm.addNestedPass<func::FuncOp>(
        //       hecate::ckks::createSCFToDpsSCFConversionPass());
        //
        //   if (opt.enable_printer)
        //     pm.addPass(createLocationSnapshotPass(
        //         OpPrintingFlags().enableDebugInfo(false, false),
        //         dir + "/" + stem + ".ckks.mlir", "ckks"));
        //   pm.addNestedPass<func::FuncOp>(hecate::ckks::createRemoveLevel());
        //   pm.addNestedPass<func::FuncOp>(hecate::ckks::createReuseBuffer());
        //   pm.addNestedPass<func::FuncOp>(hecate::ckks::createAllocLoopBuffer());
        //   pm.addNestedPass<func::FuncOp>(createCanonicalizerPass());
        //   pm.addNestedPass<func::FuncOp>(
        //       hecate::ckks::createEmitHEVM({dir + "/" + stem}));
      });
}
