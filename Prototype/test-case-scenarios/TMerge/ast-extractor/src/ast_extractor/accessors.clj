(ns ast-extractor.accessors
  (:require [clojure.spec.alpha :as s]
            [clojure.string :as str])
  (:import (com.github.javaparser.ast.stmt ExpressionStmt)
           (com.github.javaparser.ast.expr MethodCallExpr)))

(s/def ::node-type string?)
(s/def ::as-string string?)
(s/def ::resolvable boolean?)
(s/def ::scope (s/keys :req [::node-type ::as-string]))

(defn get-node-type [node] (:node-type node))
(s/fdef get-node-type
        :args (s/keys :req [::node-type])
        :ret ::node-type)

(defn get-str-repr [node] (:as-string node))
(s/fdef get-str-repr
        ::args (s/keys :req [::as-string])
        ::ret ::node-type)

(defn get-scope [node] (:scope node))
(s/fdef get-scope
        ::args (s/keys :req [::scope])
        ::ret ::scope)


(defn resolved? [node] (:resolvable node))
(s/fdef resolved?
        :args (s/keys :req [::resolvable])
        :ret ::resolvable)

(defn get-resolved-node [node] (:resolved-node node))

(defn get-return-type [node] (:return-type node))
(defn get-name [node] (:name node))
(defn get-name-str [node] (str (get-name node)))
(defn static-method? [node] (:static? node))

(defn get-method-name [node] (get-name-str node))
(defn get-expression [node] (:expression node))
(defn get-method-arguments [node] (:arguments node))
(defn get-method-parameters [node] (:parameters node))

(defn get-target [node] (:target node))
(defn get-statements [node] (:statements node))

(defn expression-statement? [node] (= "ExpressionStmt" (get-node-type node)))
(defn method-call? [node] (= "MethodCallExpr" (get-node-type node)))
(defn method-call-statement? [node] (and (expression-statement? node) (method-call? (get-expression node))))
(defn get-variables [variable-decl-statement] (:variables (get-expression variable-decl-statement)))

(defn junit-assertion? [node] (str/starts-with? (get-method-name node) "assert"))
(defn variable-declaration-expr? [node] (= "VariableDeclarationExpr" (get-node-type node)))
(defn variable-declaration-statement? [node] (and (expression-statement? node) (variable-declaration-expr? (get-expression node))))






