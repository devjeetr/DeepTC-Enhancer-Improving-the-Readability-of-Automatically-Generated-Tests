(ns summarizer.descriptors.specs
  (:require [clojure.spec.alpha :as s]))


(s/def ::descriptor-type symbol?)
(s/def ::action string?)
(s/def ::object string?)
(s/def ::object-modifier string?)
(s/def ::complement string?)
;(s/def ::statement-descriptor
;  (s/keys
;    :req [::descriptor-type ::]))

;(s/def ::action string?)
;(s/def ::method-call
;  (s/merge
;    ::base
;    (s/keys
;      ::action
;      ::object)))
