(ns summarizer.descriptor.getter-tests
  (:require [clojure.test :refer :all]
            [ast-extractor.expressions]
            [ast-extractor.extractable :refer [extract-ast-data]]
            [summarizer.descriptors.method-calls :as mc]
            [nlg.core :as nlg]
            [summarizer.helpers :as h])
  (:import (com.github.javaparser.ast.expr MethodCallExpr)))


(deftest simple-getter-test
  (let [method-call (h/construct-ast-and-get-node MethodCallExpr "Option.getSize();")
        result (mc/describe-method-call-expression (extract-ast-data method-call))]
    (is (= result "It get's Option's size."))))
