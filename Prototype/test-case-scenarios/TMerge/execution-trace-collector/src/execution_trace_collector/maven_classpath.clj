(ns execution-trace-collector.maven-classpath
  (:require [clojure.string :as str]
            [clojure.java.shell :as sh]))

; this command is used to get a classpath
(def maven-cp-command "dependency:build-classpath")

(defn class-path-line? [line]
  (and
   (pos? (count (re-seq #":" line)))
   (=
    (count (re-seq #".jar" line))
    (inc (count (re-seq #":" line))))))

(defn class-path-line [[line & remaining]]
  (if (class-path-line? line)
    line
    (class-path-line remaining)))

(defn gen-classpath [project-root]
  (class-path-line
   (str/split
    (:out (sh/sh "mvn" maven-cp-command :dir project-root))
    #"\n")))