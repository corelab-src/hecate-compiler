
#include "HecateOptMain.h"
using namespace llvm;

using namespace mlir;
using namespace hecate;

int main(int argc, char **argv) {
  // Options from MLIR Opt Main
  static cl::opt<std::string> inputFilename(
      cl::Positional, cl::desc("<input file>"), cl::init("-"));

  static cl::opt<std::string> outputFilename("o", cl::desc("Output filename"),
                                             cl::value_desc("filename"),
                                             cl::init("-"));
  // CKKS Configuration file
  static cl::opt<std::string> ckks_config{
      "ckks-config", cl::desc("Set CKKS parameters from configuration file"),
      cl::init("./config.json")};

  static HecateOptOptions opt;

  InitLLVM y(argc, argv);

  mlir::MLIRContext context;

  mlir::DialectRegistry registry;
  registry.insert<earth::EarthDialect>();
  registry.insert<ckks::CKKSDialect>();
  registry.insert<func::FuncDialect>();
  registry.insert<tensor::TensorDialect>();
  registry.insert<scf::SCFDialect>();
  registry.insert<index::IndexDialect>();
  registry.insert<transform::TransformDialect>();
  registry.insert<affine::AffineDialect>();
  registry.insert<arith::ArithDialect>();

  context.appendDialectRegistry(registry);

  // Uncomment the following to include *all* MLIR Core dialects, or
  // selectively include what you need like above. You only need to register
  // dialects that will be *parsed* by the tool, not the one generated
  // registerAllDialects(registry);
  earth::registerEarthPasses();
  ckks::registerCKKSPasses();
  hecate::registerConversionPasses();

  // Uncomment the following to make *all* MLIR core passes available.
  // This is only useful for experimenting with the command line to compose

  // Register any command line options.
  registerAsmPrinterCLOptions();
  registerMLIRContextCLOptions();
  registerPassManagerCLOptions();
  registerDefaultTimingManagerCLOptions();
  registerHecatePipeline(outputFilename, opt);
  registerPrototypePipeline(outputFilename, opt);
  // This registration must be after registering pass manager options.
  MlirOptMainConfig::registerCLOptions(registry);

  context.enableMultithreading();

  llvm::StringRef toolName("Hecate optimizer driver\n");

  // Build the list of dialects as a header for the --help message.
  std::string helpHeader = (toolName + "\nAvailable Dialects: ").str();
  {
    llvm::raw_string_ostream os(helpHeader);
    interleaveComma(registry.getDialectNames(), os,
                    [&](auto name) { os << name; });
  }
  // Parse pass names in main to ensure static initialization completed.
  cl::ParseCommandLineOptions(argc, argv, helpHeader);
  MlirOptMainConfig config = mlir::MlirOptMainConfig::createFromCLOptions();

  earth::EarthDialect::setCKKSParameters(ckks_config);

  // auto *dialect = context.getLoadedDialect<hecate::earth::EarthDialect>();
  // dialect->loadConstants(inputFilename);
  // earth::EarthDialect::loadConstants(inputFilename);

  // Set up the input file.
  std::string errorMessage;
  auto file = openInputFile(inputFilename, &errorMessage);
  if (!file) {
    llvm::errs() << errorMessage << "\n";
    return asMainReturnCode(failure());
  }

  auto output = openOutputFile(outputFilename, &errorMessage);
  if (!output) {
    llvm::errs() << errorMessage << "\n";
    return asMainReturnCode(failure());
  }

  if (failed(MlirOptMain(output->os(), std::move(file), registry, config))) {
    return asMainReturnCode(failure());
  }

  // Keep the output file if the invocation of MlirOptMain was successful.
  output->keep();
  return asMainReturnCode(success());
}
