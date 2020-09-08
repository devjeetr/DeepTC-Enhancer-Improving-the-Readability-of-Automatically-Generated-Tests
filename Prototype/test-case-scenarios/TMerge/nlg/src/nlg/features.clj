(ns nlg.features
  (:require [clojure.spec.alpha :as s]
            [nlg.constants :as c])
  (:import (simplenlg.features Feature)
           (simplenlg.framework NLGElement)))

(defn adjective-ordering [arg]
  [Feature/ADJECTIVE_ORDERING arg])
(s/fdef adjective-ordering
  :args boolean?
  :ret (s/tuple string? boolean?))

(defn aggregate-auxillary [arg]
  [Feature/AGGREGATE_AUXILIARY arg])
(s/fdef aggregate-auxillary
  :args boolean?
  :ret (s/tuple string? boolean?))

(defn complementiser [arg]
  [Feature/COMPLEMENTISER arg])
(s/def ::string-or-nlg-element
  (s/or :string string? :nlg-element #(isa? % NLGElement)))

(s/fdef complementiser
  :args ::string-or-nlg-element
  :ret (s/tuple string? ::string-or-nlg-element))

(defn conjunction [arg]
  [Feature/CONJUNCTION arg])
(s/fdef conjunction
  :args string?
  :ret (s/tuple string? string?))

;(defn conjunction-type [arg]
;      [Feature/CONJUNCTION_TYPE arg])
;(s/fdef conjunction-type
;        :args (instance? ConjunctionType))

(defn apositive [arg]
  [Feature/CONJUNCTION_TYPE arg])
(s/fdef apositive
  :args boolean?
  :ret (s/tuple string? boolean?))

(defn cue-phrase [arg]
  [Feature/CUE_PHRASE arg])
(s/fdef cue-phrase
  :args #(isa? % NLGElement)
  :ret (s/tuple string? #(isa? % NLGElement)))

(defn elided [arg]
  [Feature/ELIDED arg])
(s/fdef elided
  :args boolean
  :ret (s/tuple string? boolean))

(defn form [arg]
  [Feature/CONJUNCTION_TYPE (arg c/form-mapping)])
(s/fdef form
  :args #(contains? c/tense-mapping %)
  :ret (s/tuple string? #(contains? c/tense-mapping %)))

(defn interrogative-type [arg]
  [Feature/INTERROGATIVE_TYPE arg])
(s/fdef interrogative-type
  :args any? ; TODO
  :ret (s/tuple string? any?))

(defn is-comparative [arg]
  [Feature/IS_COMPARATIVE arg])
(s/fdef is-comparative
  :args boolean?
  :ret (s/tuple string? boolean?))

(defn is-superlative [arg]
  [Feature/IS_SUPERLATIVE arg])
(s/fdef is-superlative
  :args boolean?
  :ret (s/tuple string? boolean?))

(defn modal [arg]
  [Feature/MODAL arg])
(s/fdef modal
  :args string?
  :ret (s/tuple string? string?))

(defn negated [arg]
  [Feature/NEGATED arg])
(s/fdef negated
  :args boolean?
  :ret (s/tuple string? boolean?))

(defn number [arg]
  [Feature/NUMBER arg])
(s/fdef number
  :args any?
  :ret (s/tuple string? any?))

(defn particle [arg]
  [Feature/PARTICLE arg])
(s/fdef particle
  :args ::string-or-nlg-element
  :ret (s/tuple string? ::string-or-nlg-element))

(defn passive [arg]
  [Feature/PASSIVE arg])
(s/fdef passive
  :args boolean?
  :ret (s/tuple string? boolean?))

(defn prefect [arg]
  [Feature/PERFECT arg])
(s/fdef prefect
  :args boolean?
  :ret (s/tuple string? boolean?))

(defn person [arg]
  [Feature/PERSON arg])
(s/fdef person
  :args boolean?
  :ret (s/tuple string? boolean?))

(defn possessive [arg]
  [Feature/POSSESSIVE arg])
(s/fdef possessive
  :args boolean?
  :ret (s/tuple string? boolean?))

(defn pronomial [arg]
  [Feature/PRONOMINAL arg])
(s/fdef pronomial
  :args boolean?
  :ret (s/tuple string? boolean?))

(defn progressive [arg]
  [Feature/PROGRESSIVE arg])
(s/fdef progressive
  :args boolean?
  :ret (s/tuple string? boolean?))

(defn raise-specifier [arg]
  [Feature/RAISE_SPECIFIER arg])
(s/fdef raise-specifier
  :args boolean?
  :ret (s/tuple string? boolean?))

(defn suppress-genitive-in-gerund [arg]
  [Feature/SUPPRESS_GENITIVE_IN_GERUND arg])
(s/fdef suppress-genitive-in-gerund
  :args boolean?
  :ret (s/tuple string? boolean?))

(defn supressed-complementiser [arg]
  [Feature/SUPRESSED_COMPLEMENTISER arg])
(s/fdef supressed-complementiser
  :args boolean?
  :ret (s/tuple string? boolean?))

(defn tense [arg]
  [Feature/TENSE (arg c/tense-mapping)])
(s/fdef tense
  :args #(contains? c/tense-mapping %)
  :ret (s/tuple string? boolean?))

