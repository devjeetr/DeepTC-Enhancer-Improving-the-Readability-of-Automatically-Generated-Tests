(ns nlg.core
  (:require [clojure.spec.alpha :as s]
            [clojure.spec.test.alpha :as stest]
            [nlg.helpers :as h]
            [nlg.specs :as nlg-specs]
            [nlg.features :as f])
  (:import (simplenlg.framework NLGFactory CoordinatedPhraseElement)
           (simplenlg.lexicon Lexicon)
           (simplenlg.features Feature Tense Form)
           (simplenlg.realiser.english Realiser)
           (simplenlg.phrasespec SPhraseSpec)))

(def lexicon (Lexicon/getDefaultLexicon))
(def factory (NLGFactory. lexicon))
(def realizer (Realiser. lexicon))


; Core API

(defn clause [& args]
  (->> (apply merge-with into args)
       (h/create-clause factory)))


(defn subject [phrases & args]
  (if (coll? phrases)
    (-> (apply hash-map args)
        (select-keys [:features])
        (assoc :phrases phrases)
        (->> (h/create-coordinated-phrase factory)
             (hash-map :subject)))

    {:subject phrases}))

(defn verb [arg]
  {:verb arg})

(defn object [args]
  {:object args})

(defn compl [args]
  {:complement args})

(defn verb-phrase [word & args]
  (-> (apply hash-map args)
      (assoc :verb word)
      (->> (h/verb-phrase factory))))

(defn noun-phrase [word & args]
  (-> (apply hash-map args)
      (assoc :noun word)
      (->> (h/noun-phrase factory))))

(defn preposition-phrase [word & args]
  (-> (apply hash-map args)
      (assoc :preposition word)
      (->> (h/prepositional-phrase factory))))

(defn feature [& args]
  [:features (->> (flatten args)
                  (partition 2)
                  (map (fn [[feature-name feature-value]]
                         {:feature-name feature-name :feature-value feature-value})))])

(defn realise [nlg-element]
  (.realise realizer nlg-element))

(defn realise-sentence [nlg-element]
  (.realiseSentence realizer nlg-element))

(stest/instrument 'h)
;
;(clause
;  (subject "Mary")
;  (object (noun-phrase "monkey" :determiner "the"))
;  (verb "chase"))
