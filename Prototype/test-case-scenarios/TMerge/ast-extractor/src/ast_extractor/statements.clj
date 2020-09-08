(ns ast-extractor.statements
  (:require [ast-extractor.extractable :refer [extractable extract-ast-data]]
            [ast-extractor.common :refer [extract-default-fields]]
            [ast-extractor.expressions]
            [javaparser-wrapper.core :as j]
            [clojure.spec.alpha :as s])
  (:import (com.github.javaparser.ast.body MethodDeclaration)
           (com.github.javaparser.ast.stmt ForStmt BlockStmt ExpressionStmt TryStmt ForEachStmt WhileStmt IfStmt ReturnStmt)))


(extend-type BlockStmt
  extractable
  (extract-ast-data [^BlockStmt statement]
    (->> (.getStatements statement)
         (map extract-ast-data)
         (j/node-list->seq))))

(extend-type ExpressionStmt
  extractable
  (extract-ast-data [^ExpressionStmt stmt]
    (extract-default-fields
      stmt
      {:expression (extract-ast-data (.getExpression stmt))})))

(extend-type MethodDeclaration
  extractable
  (extract-ast-data [^MethodDeclaration method-declaration]
    (extract-default-fields
      method-declaration
      {:statements (map extract-ast-data (j/get-method-statements method-declaration))})))

(extend-type ForStmt
  extractable
  (extract-ast-data [^ForStmt for-stmt]
    (extract-default-fields
      for-stmt
      {:initialization (-> (.getInitialization for-stmt)
                           (j/node-list->seq)
                           (map extract-ast-data))
       :comparison     (let [comparison (j/get-optional (.getCompare for-stmt))]
                         (when-not (nil? comparison) (extract-ast-data comparison)))
       :update         (-> (.getUpdate for-stmt)
                           (j/node-list->seq)
                           (map extract-ast-data))
       :body           (extract-ast-data (.getBody for-stmt))})))

(extend-type TryStmt
  extractable
  (extract-ast-data [^TryStmt try-stmt]
    (extract-default-fields
      try-stmt
      {})))                                                 ; TODO

(extend-type ForEachStmt
  extractable
  (extract-ast-data [^ForEachStmt for-each-stmt]
    (extract-default-fields for-each-stmt {})))

(extend-type WhileStmt
  extractable
  (extract-ast-data [^WhileStmt while-stmt]
    (extract-default-fields while-stmt {})))                ;TODO

(extend-type IfStmt
  extractable
  (extract-ast-data [^IfStmt if-stmt]
    (extract-default-fields if-stmt {})))

;(extend-type ReturnStmt
;  extractable
;  (extract-ast-data [^ReturnStmt return-stmt]
;    (extract-default-fields return-stmt {})))