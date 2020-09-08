(ns summarizer.descriptors.generator
  (:require [ast-extractor.accessors :as a]
            [taoensso.timbre :as timbre]
            [summarizer.descriptors.method-calls :refer [generate-method-call-descriptor]]
            [summarizer.descriptors.variable-declaration :refer [generate-variable-declaration-descriptor]]))

(defn generate-expression-summary-descriptor [statement]
  (let [expression (a/get-expression statement)]
    (cond
      (a/method-call? expression) (generate-method-call-descriptor expression)
      (a/variable-declaration-expr? expression) (generate-variable-declaration-descriptor expression)
      :else (doall (timbre/warn "Expression statement type not supported:" (a/get-node-type expression))))))

(defn generate-summary-descriptor-for-statement [statement]
  (cond
    (a/expression-statement? statement) (generate-expression-summary-descriptor statement)
    :else (timbre/warn "Statement type not supported:" (a/get-node-type statement))))