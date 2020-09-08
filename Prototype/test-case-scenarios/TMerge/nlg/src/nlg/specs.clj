(ns nlg.specs
  (:require [clojure.spec.alpha :as s]))


; specs


(s/def ::noun string?)
(s/def ::determiner string?)
(s/def ::pre-modifier string?)
(s/def ::post-modifier string?)
(s/def ::feature-name string?)
(s/def ::feature-value any?)
(s/def ::feature (s/keys :req [::feature-name ::feature-value]))
(s/def ::features (s/coll-of ::feature))

(s/def ::noun-phrase-descriptor (s/keys :req [::noun]
                                        :opt [::determiner
                                              ::pre-modifier
                                              ::post-modifier
                                              ::features]))

(s/def ::verb string?)
(s/def ::object any?)
(s/def ::indirect-object any?)
(s/def ::verb-phrase-descriptor (s/keys :req [::verb]
                                        :opt [::pre-modifier
                                              ::post-modifier
                                              ::object
                                              ::indirect-object
                                              ::features]))
(s/def ::phrase-spec any?)                                  ; TODO
