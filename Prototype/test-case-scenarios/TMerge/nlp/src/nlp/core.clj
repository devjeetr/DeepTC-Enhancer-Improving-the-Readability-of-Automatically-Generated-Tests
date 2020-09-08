(ns nlp.core
  (:require [opennlp.nlp :refer [make-pos-tagger]]
            [clojure.string :as str]
            [clojure.java.io :as io]
            [org.clojurenlp.core :refer [pos-tag]])

  (:import [org.apache.commons.lang3 StringUtils]))


(defn get-pos-tags [words]
  (->> (pos-tag words)
       (map #(.tag %))
       (map vector words ,)))

(defn camel-case-split
  [identifier]
  (seq (StringUtils/splitByCharacterTypeCamelCase identifier)))

(defn identifier-tokens
  [identifier]
  (map str/lower-case (camel-case-split identifier)))

(defn create-tag-str [tagged]
  (str/join "#" (map second tagged)))

(defn tag-str-from-method-name [method-name]
  (create-tag-str (get-pos-tags (identifier-tokens method-name))))

(def common-abbreviations {"args" "arguments"
                           "arg"  "argument"})

(defn expand-identifier [identifier]
  (let [expansion (common-abbreviations (str/lower-case identifier))]
    (if (nil? expansion)
      identifier
      expansion)))
;
(defn expand-identifiers [identifiers]
  (map expand-identifier identifiers))
