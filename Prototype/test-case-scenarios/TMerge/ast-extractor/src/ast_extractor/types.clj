(ns ast-extractor.types
  (:require [javaparser-wrapper.core :as j]
            [ast-extractor.extractable :refer [extractable extract-ast-data]]
            [ast-extractor.common :refer [extract-default-fields get-parameters get-arguments] :as c]
            [clojure.spec.alpha :as s])
  (:import (com.github.javaparser.ast.type Type)
           (javax.lang.model.type PrimitiveType)))


;; extractors
(extend-type Type
  extractable
  (extract-ast-data [^Type expr]
    (extract-default-fields
      expr
      {:type-name (str expr)})))

(extend-type PrimitiveType
  extractable
  (extract-ast-data [^PrimitiveType expr]
    (extract-default-fields
      expr
      {})))                                                 ;TODO