(ns ast-extractor.extractable
  (:require [taoensso.timbre :as timbre]))

(defprotocol extractable
  (extract-ast-data [x] "extracts data to an object"))



; misc
(extend-type nil
  extractable
  (extract-ast-data [node]
    (timbre/trace node)))