(ns summarizer.aggregator-tests
  (:require [clojure.test :refer :all]
            [summarizer.helpers :as helpers]
            [javaparser-wrapper.core :as j]
            [ast-extractor.statements]
            [ast-extractor.types]
            [ast-extractor.expressions]
            [ast-extractor.extractable :refer [extract-ast-data]]
            [summarizer.aggregation.base :as a]
            [summarizer.aggregation.common :as cc]
            [ast-extractor.accessors :as acc]
            [taoensso.timbre :as timbre])
  (:import (com.github.javaparser.ast.body MethodDeclaration)))

(timbre/merge-config! {:level :fatal})

(deftest same-sequential-action-aggregation-test
  (let [declaration (helpers/construct-ast-and-get-node MethodDeclaration "Option x = new Option();
                                     x.setOption(\"hello\");
                                     x.setOption(\"sofar\");")
        statements (acc/get-statements (extract-ast-data declaration))
        result (a/sequential-same-action-aggregator statements)]
    (is (= (first result) (first statements)) "statements are equal")
    (is (acc/get-node-type (second result)) :aggregated)
    (is (:aggregation (second result)) :sequential-same-action)
    (is (:objects (second result)) (set ["x"]))))
;
(deftest same-sequential-action-aggregation-test-shouldnt-aggregate-last
  (let [declaration (helpers/construct-ast-and-get-node MethodDeclaration "Option x = new Option();
                                      Option y = new Option();
                                     x.setOption(\"hello\");
                                     ")
        statements (acc/get-statements (extract-ast-data declaration))
        result (a/sequential-same-action-aggregator statements)]
    (is (= (last result) (last statements)) "statements are equal")
    (is (acc/get-node-type (first result)) :aggregated)
    (is (:aggregation (first result)) :sequential-same-action)
    (is (:objects (first result)) (set ["x"]))))

(deftest same-sequential-object-aggregation-test
  (let [declaration (helpers/construct-ast-and-get-node MethodDeclaration "Option x = new Option();
                                     x.setOption(\"hello\");")
        statements (acc/get-statements (extract-ast-data declaration))
        result (a/sequential-same-object-aggregator statements)]
    (is (= (acc/get-node-type (first result)) :aggregated))
    (is (:aggregation (first result)) :sequential-same-object)
    (is (:actions (first result)) (set [:variable-declaration "setOption"]))))


(deftest same-sequential-object-and-same-sequential-action-test
  (let [declaration (helpers/construct-ast-and-get-node MethodDeclaration "Option x = new Option();
                                    Option y = new Option();
                                     x.setOption(\"hello\");
                                     y.setOption(22);")
        statements (acc/get-statements (extract-ast-data declaration))
        aggregators (comp  a/sequential-same-object-aggregator
                           a/sequential-same-action-aggregator)
        result (aggregators statements)]
    (is (= (acc/get-node-type (first result)) :aggregated))
    (is (= :sequential-same-object (:aggregation (first result))))
    (is (=  ["x" "y"] (:objects (first result))))
    (is (:objects (first result)) (set ["x" "y"]))))

;
;(deftest consecutive-reducer-test
;  (let [result (transduce (cc/consecutive-reducer #(every? even? [%1 %2]) (fn [x] x))
;                          conj
;                          [2 2 3])]
;    (is (= (second result) 3))))

(run-tests 'summarizer.aggregator-tests)