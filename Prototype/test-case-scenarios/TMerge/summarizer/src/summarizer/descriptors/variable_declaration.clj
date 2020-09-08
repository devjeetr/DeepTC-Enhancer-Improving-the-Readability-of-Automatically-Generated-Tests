(ns summarizer.descriptors.variable-declaration
  (:require [ast-extractor.expressions :as expr]
            [clojure.spec.alpha :as s]
            [summarizer.common :refer [describe-as-value format-list-output]]
            [ast-extractor.accessors :as a]))

(defn describe-single-variable [{:keys [type name]}]
  {:type     :single-variable-declaration
   :action   "instantiate"
   :object   [(a/get-str-repr type) name]
   :seconday nil}) ;TODO

(s/fdef describe-single-variable
  :args ::expr/variable-declarator
  :ret string?)

(defn describe-variables [nodes]
  (format-list-output (map describe-single-variable nodes)))
(s/fdef describe-variables
  :args (s/coll-of ::expr/variable-declarator)
  :ret string?)

(defn generate-variable-declaration-descriptor [{:keys [variables]}]
  "
    Describes a VariableDeclarationExpr
  "
  (cond
    (= 1 (count variables)) (describe-single-variable (first variables))
    (> 1 (count variables)) (describe-variables variables)))

(s/fdef generate-variable-declaration-descriptor
  :args ::expr/variable-declaration-expr
  :ret string?)