(ns summarizer.aggregation.aggregators
  (:require [clojure.string :as str])
  (:import (com.github.javaparser.ast CompilationUnit)))


; TODO


(defn potential-missing-assertion-aggregator
  "
    Suggests potential missing assertions
  "
  [statements ^CompilationUnit cu])

(defn equivalent-assertions [statements ^CompilationUnit cu])
(defn inconsequential-statements [statements ^CompilationUnit cu])
(defn evo-suite-exception-aggregator [statements ^CompilationUnit cu])

(defn get-action
  [statement]
  (:name statement))

(defn resolve-scope [scope]
  (cond
    (= (:node-type scope) "NameExpr") (:asString scope)
    (= (:node-type scope) "CastExpr") (resolve-scope (:target scope))
    (= (:node-type scope) "EnclosedExpr") (resolve-scope (:inner scope))
    :else (resolve-scope (:scope scope))))

(def single-argument-asserts #{"assertTrue" "assertFalse"
                               "assertNull" "assertNotNull"})
(defn get-object-of-assertion
  [method-call]
  (resolve-scope (last (:arguments method-call))))

(defn get-object
  [statement]
  (cond
    (= "aggregated" (:node-type statement)) (:object statement)
    (str/starts-with? (:name (:expression statement)) "assert") (get-object-of-assertion (:expression statement))
    :else (resolve-scope (:expression statement))))

(defn method-call? [statement]
  (every?
   true?
   [(= (:node-type statement) "ExpressionStmt")
    (= (:node-type (:expression statement)) "MethodCallExpr")]))

(defn same-action? [a b]
  (every?
   true?
   [(method-call? a)
    (method-call? b)
    (= (get-action (:expression a)) (get-action (:expression b)))]))

(defn same-object? [a b]
  (and
   (= (get-object a) (get-object b))))

(defn assertion?
  [statement]
  (and (method-call? statement)
       (str/starts-with? (:name (:expression statement)) "assert")))

(def sequential-same-object-reducer
  (partial
   consecutive-reducer
   #(and
     (method-call? %1)
     (method-call? %2)
     (same-object? %1 %2)
     (not (same-action? %1 %2))
     (and (not (assertion? %1))
          (not (assertion? %2))))))

(defn sequential-same-object
  [statements]
  (map
   #(if (sequential? %)
      {:node-type    "aggregated"
       :aggregation "same-object-sequential"
       :statements  %
       :object      (get-object (first %))}
      %)
   (reduce
    sequential-same-object-reducer
    statements)))

(def sequential-same-action-reducer
  (partial
   consecutive-reducer
   #(and
     (method-call? %1)
     (method-call? %2)
     (same-object? %1 %2)
     (same-action? %1 %2))))

(defn sequential-same-action-before-assert
  "
    list.add(2)
    list.add(3)
    assert(list.length, 2)
  "
  [statements]
  (map
   #(if (sequential? %)
      {:node-type    "aggregated"
       :aggregation "same-action-sequential"
       :statements  %
       :object      (get-object (first %))}
      %)
   (reduce
    sequential-same-action-reducer
    statements)))

(def sequential-assertion-reducer
  (partial
   consecutive-reducer
   #(and
     (assertion? %1)
     (assertion? %2)
     (same-object? %1 %2))))

(defn sequential-assertions
  [statements]
  (map
   #(if (sequential? %)
      {:node-type    "aggregated"
       :aggregation "same-object-assertion-sequential"
       :statements  %
       :object      (get-object (first %))}
      %)
   (reduce
    sequential-assertion-reducer
    statements)))

(def action-assertion-aggregator-reducer
  (partial
   consecutive-reducer
   #(and
     (or                                                  ; first statement needs to be
      (and
       (not (assertion? %1))                            ; either a method call
       (method-call? %1))                               ; (but not assertion)
      (= "aggregated" (:node-type %1)))                   ; or an aggregated node
     (or
      (assertion? %2)
      (and
       (= "aggregated" (:node-type %2))
       (= "same-object-assertion-sequential" (:aggregation %2))))
     (same-object? %1 %2))))

(defn action-assertion-aggregator
  [statements]
  (map
   #(if (sequential? %)
      {:node-type    "aggregated"
       :aggregation "action-assertion-aggregate"
       :statements  %
       :object      (get-object (first %))}
      %)
   (reduce
    action-assertion-aggregator-reducer
    statements)))

(def aggregate-all
  (comp
   action-assertion-aggregator
   sequential-same-action-before-assert
   sequential-same-object
   sequential-assertions))
