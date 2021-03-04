//===- TestUserangeAnalysis.cpp - Test userange construction and information
//-------===//
//
// Part of the LLVM Project, under the Apache License v2.0 with LLVM Exceptions.
// See https://llvm.org/LICENSE.txt for license information.
// SPDX-License-Identifier: Apache-2.0 WITH LLVM-exception
//
//===----------------------------------------------------------------------===//
//
// This file contains test passes for constructing and resolving userange
// information.
//
//===----------------------------------------------------------------------===//

#include "mlir/Analysis/BufferAliasAnalysis.h"
#include "mlir/Analysis/UserangeAnalysis.h"
#include "mlir/Pass/Pass.h"
#include "mlir/Transforms/BufferUtils.h"

using namespace mlir;

namespace {

struct TestUserangePass : public PassWrapper<TestUserangePass, FunctionPass> {
  void runOnFunction() override {
    llvm::errs() << "Testing : " << getFunction().getName() << "\n";
    UserangeAnalysis(getFunction(), BufferPlacementAllocs(getFunction()),
                     BufferAliasAnalysis(getFunction()))
        .print(llvm::errs());
  }
};

} // end anonymous namespace

namespace mlir {
namespace test {
void registerTestUserangePass() {
  PassRegistration<TestUserangePass>(
      "test-print-userange",
      "Print the contents of a constructed userange analysis.");
}
} // namespace test
} // namespace mlir
