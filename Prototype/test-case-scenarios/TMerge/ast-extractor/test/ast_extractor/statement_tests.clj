(ns ast-extractor.statement-tests
  (:require [clojure.test :refer :all]
            [ast-extractor.helpers :as helpers]
            [ast-extractor.statements]
            [ast-extractor.extractable :refer [extract-ast-data]]
            [javaparser-wrapper.core :as j])
  (:import (com.github.javaparser.ast.stmt ExpressionStmt)))


(deftest expression-statement-basic-test
  (let [cu (helpers/parse-statement "File x = new File(\"./data/\");")
        expr (first (j/find-all ExpressionStmt cu))
        result (extract-ast-data expr)]
    (is (= (:node-type result) "ExpressionStmt"))
    (is (= (:node-type (:expression result)) "VariableDeclarationExpr"))))