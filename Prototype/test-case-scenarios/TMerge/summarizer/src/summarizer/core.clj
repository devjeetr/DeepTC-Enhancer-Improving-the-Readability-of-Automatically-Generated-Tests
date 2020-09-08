(ns summarizer.core
  (:require [javaparser-wrapper.core :as j]
            [ast-extractor.expressions :as expr-extractor]
            [ast-extractor.statements :as stmt-extractor]
            [summarizer.descriptors.generator :as g]
            [summarizer.summarizer :refer [summarize]]
            [clojure.string :as str]
            [taoensso.timbre :as timbre]
            [ast-extractor.accessors :as a]
            [clojure.spec.alpha :as s]
            [nlg.core :as c]
            [nlg.features :as features]))



(defn describe-test-method [test-method-declaration]
  (let [statements (a/get-statements test-method-declaration)]
    (map (comp summarize g/generate-summary-descriptor-for-statement) statements)))

(defn describe-statement [statement]
  (->> (g/generate-summary-descriptor-for-statement statement)
       (summarize)))

