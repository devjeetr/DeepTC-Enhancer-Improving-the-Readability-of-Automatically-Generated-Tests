(ns summarizer.aggregation.base
  (:require [summarizer.aggregation.common :as c]
            [ast-extractor.accessors :as acc]))



(defn method-call-or-var-dec? [arg]
  (or (acc/method-call-statement? arg)
      (acc/variable-declaration-statement? arg)
      (c/aggregated-statement? arg)))




(defn map-if [predicate? transform-item coll]
  (map (fn [item]
         (if (predicate? item)
           (transform-item item)
           item))
       coll))

(defn construct-aggregator [reducer aggregation-transformer]
  (fn [statements]
    (->> statements
         (reduce reducer)
         (map-if sequential? aggregation-transformer))))

(def sequential-same-action-reducer
  (c/consecutive-reducer #(and
                            (method-call-or-var-dec? %1)
                            (method-call-or-var-dec? %2)
                            (c/same-action? %1 %2))))
(def sequential-same-action-aggregator
  "
    Returns a transducer that aggregates consecutive
    same action statements
  "
  (construct-aggregator
    sequential-same-action-reducer
    (fn [statements]
      {:node-type   :aggregated
       :aggregation :sequential-same-action
       :statements  statements
       :actions     (flatten [(c/get-action (first statements))])
       :objects     (flatten (map c/get-object statements))})))

(def sequential-same-object-reducer
  (c/consecutive-reducer
    #(and
       (method-call-or-var-dec? %1)
       (method-call-or-var-dec? %2)
       (c/same-object? %1 %2))))
(def sequential-same-object-aggregator
  "
    Returns a transducer that aggregates consecutive
    same action statements
  "
  (construct-aggregator
    sequential-same-object-reducer
    (fn [statements]
      {:node-type   :aggregated
       :aggregation :sequential-same-object
       :statements  statements
       :actions     (flatten (map c/get-action statements))
       :objects     (flatten [(c/get-object (first statements))])})))





