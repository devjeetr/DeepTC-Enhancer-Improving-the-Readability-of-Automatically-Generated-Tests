(ns ast-extractor.common
  (:require [javaparser-wrapper.core :as j]
            [ast-extractor.extractable :refer [extract-ast-data]]

            [clojure.spec.alpha :as s])

  (:import (com.github.javaparser.ast Node)))

(s/def ::as-string string?)
(s/def ::node-type string?)
(s/def ::base-node (s/keys ::as-string ::node-type))
(s/def ::java-parser-node #(instance? Node %))
(s/def ::any-map map?)

(defn extract-default-fields
  [node descriptor]
  "
    Extractors node-type and string representation
    of the given node
  "
  (assoc
   descriptor
   :node-type (j/get-meta-model-str node)
   :as-string (str node)))
(s/fdef extract-default-fields
        :args (s/cat ::java-parser-node ::any-map)
        :ret ::base-node)

(defn get-arguments [expr]
  "
    Helper function to get arguments
    of a method call or constructor
    invocation. If there are no arguments,
    returns nil.
  "
  (.getArguments expr))

(defn get-parameters [expr]
  "
    Helper function to get parameters
    of a method declaration. If expr is nil
    or there are no parameters for expr,
    returns nil.
  "
  (when (and (not (nil? expr)) (pos? (count (.getParameters expr)))) (map j/get-name-str (.getParameters expr))))
