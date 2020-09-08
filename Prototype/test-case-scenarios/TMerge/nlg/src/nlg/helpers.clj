(ns nlg.helpers
  (:require [clojure.spec.alpha :as s]
            [nlg.specs :as specs])
  (:import (simplenlg.framework NLGFactory CoordinatedPhraseElement)
           (simplenlg.lexicon Lexicon)
           (simplenlg.features Feature Tense Form)
           (simplenlg.realiser.english Realiser)
           (simplenlg.phrasespec SPhraseSpec)))




; helpers


(defn- set-subject! [phrase subject]
  (when subject
    (.setSubject phrase subject)))

(defn- set-object! [phrase object]
  (when object
    (.setObject phrase object)))

(defn- set-verb! [phrase verb]
  (when verb
    (.setVerb phrase verb)))

(defn set-indirect-object! [phrase object]
  (when object
    (.setIndirectObject phrase object)))

(defn set-feature! [phrase {:keys [feature-name feature-value]}]
  (.setFeature phrase feature-name feature-value))
(s/fdef set-feature!
  :args (s/cat ::specs/phrase-spec
               ::specs/feature))

(defn set-features! [phrase features]
  (doseq [feature features]
    (set-feature! phrase feature)))

(s/fdef set-features!
  :args (s/cat ::specs/phrase-spec ::specs/features))

(defn create-feature [name value]
  {:feature-name name :feature-value value})

(defn- create-noun-phrase
  [factory noun] (.createNounPhrase factory noun))

(defn- set-post-modifier! [phrase modifier]
  (when modifier
    (.setPostModifier phrase modifier)))

(defn- set-pre-modifier! [phrase modifier]
  (when modifier
    (.setPreModifier phrase modifier)))

(defn- set-determiner! [phrase determiner]
  (when determiner
    (.setDeterminer phrase determiner)))

(defn- set-complement! [clause complement]
  (when complement
    (.setComplement clause complement)))

(defn noun-phrase [factory {:keys [noun features post pre determiner]
                            :or   {features []}}]
  (let [np (create-noun-phrase factory noun)]
    (set-features! np features)
    (set-post-modifier! np post)
    (set-pre-modifier! np pre)
    (set-determiner! np determiner)
    np))
(s/fdef noun-phrase
  :args ::specs/noun-phrase-descriptor
  :ret any?)                                          ;TODO


(defn verb-phrase [factory {:keys [verb features post pre object indirect-object]
                            :or   {features []}}]
  (let [vp (.createVerbPhrase factory verb)]
    (set-features! vp features)
    (set-post-modifier! vp post)
    (set-pre-modifier! vp pre)
    (set-object! vp object)
    (set-indirect-object! vp indirect-object)
    vp))
(s/fdef verb-phrase
  :args ::specs/verb-phrase-descriptor
  :ret any?)                                          ;TODO

(defn prepositional-phrase [factory {:keys [preposition features post pre complement object]}]
  (let [pp (.createPrepositionPhrase factory preposition)]
    (set-features! pp features)
    (set-post-modifier! pp post)
    (set-pre-modifier! pp pre)
    (set-complement! pp complement)
    (set-object! pp object)
    pp))

(defn add-coordinates! [cp phrases]
  (doseq [phrase phrases]
    (.addCoordinate cp phrase)))

(defn create-coordinated-phrase [factory {:keys [phrases features] :or {features []}}]
  (let [cp (.createCoordinatedPhrase factory)]
    (add-coordinates! cp phrases)
    (set-features! cp features)
    cp))


(defn create-clause [factory {:keys [subject object verb features complement]
                              :or   {features []}}]
  (let [clause (.createClause factory)]
    (set-subject! clause subject)
    (set-object! clause object)
    (set-verb! clause verb)
    (set-features! clause features)
    (set-complement! clause complement)
    clause))

