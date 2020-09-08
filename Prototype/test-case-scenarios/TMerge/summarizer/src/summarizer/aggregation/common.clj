(ns summarizer.aggregation.common
  (:require [clojure.spec.alpha :as s]
            [ast-extractor.accessors :as acc]))

(def not-sequential? (complement sequential?))

(defn consecutive-reducer
  ([predicate?] (fn [acc curr]
                  (consecutive-reducer predicate? acc curr)))
  ([predicate? acc curr]
   (cond
     (not (sequential? acc)) (if (predicate? acc curr)
                               [[acc curr]] [acc curr])
     (not (sequential? (last acc))) (if (predicate? (last acc) curr)
                                      (conj (pop acc) [(last acc) curr])
                                      (conj acc curr))
     :else (if (predicate? (last (last acc)) curr)
             (conj (pop acc) (conj (last acc) curr))
             (conj acc curr)))))

(defn aggregated-statement? [statement]
  (= :aggregated (:node-type statement)))

(defn get-action-for-method-call [statement]
  (acc/get-method-name (acc/get-expression statement)))

(defn get-action-for-variable-declaration-statement [statement]
  :variable-declaration)

(defn get-action-for-aggregated-statement [statement]
  (when (not (nil? (:actions statement)))
    (:actions statement)))

(defn get-action [statement]
  (cond
    (acc/method-call-statement? statement) (get-action-for-method-call statement)
    (acc/variable-declaration-statement? statement) (get-action-for-variable-declaration-statement statement)
    (aggregated-statement? statement) (get-action-for-aggregated-statement statement)
    :else (throw (ex-info "No action defined for statement type" {:node-type (acc/get-node-type statement)}))))

(defn same-action? [a b]
  (= (set (flatten [(get-action a)]))
     (set (flatten [(get-action b)]))))

(defn get-object-for-variable-declaration-statement [statement]
  (acc/get-name-str (first (acc/get-variables statement))))

(defn get-root-scope [method-call]
  (if (nil? (acc/get-scope method-call))
    (acc/get-str-repr method-call)
    (get-root-scope (acc/get-scope method-call))))

(defn get-object-for-method-call-statement [statement]
  (get-root-scope (acc/get-expression statement)))

(defn get-object-for-aggregated-statement [statement]
  (when (not (nil? (:objects statement)))
    (:objects statement)))

(defn get-object [statement]
  (cond
    (acc/variable-declaration-statement? statement) (get-object-for-variable-declaration-statement statement)
    (acc/method-call-statement? statement) (get-object-for-method-call-statement statement)
    (aggregated-statement? statement) (get-object-for-aggregated-statement statement)
    :else (throw (ex-info "No object defined for statement type" {:node-type (acc/get-node-type statement)}))))

(defn same-object? [a b]
  (= (set (flatten [(get-object a)]))
     (set (flatten [(get-object b)]))))

