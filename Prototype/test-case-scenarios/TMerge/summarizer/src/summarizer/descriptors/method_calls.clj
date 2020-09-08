(ns summarizer.descriptors.method-calls
  (:require [ast-extractor.accessors :as a]
            [ast-extractor.expressions :as expr]
            [taoensso.timbre :as timbre]
            [nlp.core :as nlp]
            [clojure.spec.alpha :as s]
            [clojure.string :as str]
            [nlg.core :as nlg]
            [nlg.features :as nf]
            [summarizer.descriptors.junit-assertions :refer [describe-junit-assertion]]))

(defn setter? [node]
  (str/starts-with? (a/get-method-name node) "set"))

(defn getter? [node]
  (str/starts-with? (a/get-method-name node) "get"))

(defn vb-nn? [node]
  (= (nlp/tag-str-from-method-name (a/get-method-name node))
     "VB#NN"))
(defn vb? [node]
  (= (nlp/tag-str-from-method-name (a/get-method-name node))
     "VB"))

(def name-tokens (comp nlp/identifier-tokens a/get-method-name))

(defn- generate-setter-descriptor [node]
  (let [tokens (name-tokens node)
        scope (a/get-str-repr (a/get-scope node))]
    {:type "setter"
     :action "set"
     :object [scope (second tokens)]
     :seconday nil}))

(defn- generate-getter-descriptor [node]
  (let [tokens (name-tokens node)
        scope (a/get-str-repr (a/get-scope node))]
    {:type "getter"
     :action "get"
     :object [scope (second tokens)]
     :secondary nil}))


(defn- generate-vb-nn-descriptor [node]
  (let [tokens (name-tokens node)
        scope (a/get-str-repr (a/get-scope node))]
    {:type "VB#NN"
     :action (first tokens)
     :object [scope (second tokens)]
     :secondary nil})) ;TODO

(defn- generate-vb-descriptor [node]
  (let [tokens (name-tokens node)
        scope (a/get-str-repr (a/get-scope node))]
    {:type "NN"
     :action (first tokens)
     :object (if (pos? (count (a/get-method-arguments node)))
                 (first (a/get-str-repr (a/get-method-arguments node)))
                 scope)
     :secondary nil})) ;TODO

(defn- gen-degenerate-descriptor [node]
  {:action "call"
   :object (a/get-method-name node)
   :receiver (a/get-str-repr (a/get-scope node))})

(defn generate-method-call-descriptor [node]
  (cond
    (setter? node) (generate-setter-descriptor node)
    (getter? node) (generate-getter-descriptor node)
    (vb-nn? node) (generate-vb-nn-descriptor node)
    (vb? node) (generate-vb-descriptor node)
    :else (gen-degenerate-descriptor node)))


(defn describe-method-call-expression [node]
  (cond
    (not (a/junit-assertion? node))  (generate-method-call-descriptor node)
    :else (describe-junit-assertion node)))

(s/fdef describe-method-call-expression
  :args ::expr/arguments
  :ret string?)
