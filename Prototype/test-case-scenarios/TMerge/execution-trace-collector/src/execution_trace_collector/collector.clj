(ns execution-trace-collector.collector
  (:require [clojure.string :as str]
            [clojure.java.shell :as sh]
            [javaparser-wrapper.core :as j]
            [clojure.java.io :as io]
            [clojure.spec.alpha :as s]
            [taoensso.timbre :as timbre]
            [execution-trace-collector.maven-classpath :refer [gen-classpath]])
  (:import (java.io File)))

(defn join-path [& args]
  (.getPath (apply io/file args)))

(defn get-all-classfiles [^File dir]
  " returns a list of all files that end with
    .class in the given directory
  "
  (let [all-files (file-seq dir)]
    (filter #(str/ends-with? % ".class") all-files)))

(defn path->classname [filename]
  "
    converts path to fully qualified class name
    org/apache/Option.class -> org.apache.Option
  "
  (-> filename
      (str/replace #"/" ".")
      (str/replace #"\.class" "")))

(defn classes-in-dir [dir]
  "
    returns a list of of classes in a given directory.
    For example if we have the following directory structure:
      target
        /test-classes
          org/apache/Option.class
          org/apache/Builder.class
    returns:
      [org.apache.Option org.apache.Builder]
  "
  (->> (io/file dir)
       (get-all-classfiles)
       (map #(.toString %))
       (map #(str/replace % dir ""))
       (map #(str/replace-first % #"/" ""))
       (map path->classname)))

(defn scaffolding? [^String class-name]
  (not (nil? (re-find #"^(?:[a-zA-Z]+\.)+[a-zA-Z]+_ESTest_scaffolding$" class-name))))

(def not-scaffolding? (complement scaffolding?))

(defn partial-class? [^String class-name]
  (not (nil? (re-find #"^(?:\w+\.)+(\w+)\$.+$" class-name))))

(def not-partial-class? (complement partial-class?))

(defn test-class? [^String class-name]
  (->> ((juxt not-partial-class? not-scaffolding?) class-name)
       (every? true? ,)
       ))

(defn get-test-classes [classes]
  (filter test-class? classes))

(def junit-cmd "org.junit.runner.JUnitCore")
(def mapping {:classes-to-instrument "c"
              :output                "o"
              :tests-to-include      "i"
              :test-class            "t"})

(defn gen-agent-args [args]
  (str/join " " (map #(str/join ["-" ((first %) mapping) (second %)]) args)))

(defn gen-agent-cmd [args]
  (str/join ["-javaagent:"
             (:agent-jar args)
             "="
             (gen-agent-args (dissoc args :agent-jar))]))

(defn get-exec-trace-shell-cmd [{:keys [test-class classpath], :as args}]
  (let [agent-keys [:agent-jar
                    :classes-to-instrument
                    :test-class
                    :tests-to-include
                    :output]
        agent-cmd (gen-agent-cmd
                   (select-keys args
                                agent-keys))]
    ["java" "-classpath" classpath agent-cmd junit-cmd test-class]))

(defn run-execution-trace-collector [args]
  (let [shell-cmd (get-exec-trace-shell-cmd args)]
    (timbre/info "Running shell command")
    (timbre/debug (str/join " " shell-cmd))
    (apply sh/sh shell-cmd)))

(defn get-test-name [^String test-class]
  (let [matcher (re-matcher #"^(?:[a-zA-Z]+\.)+(\w+)$" test-class)]
    (println test-class)
    (re-find matcher)
    (second (re-groups matcher))))

(defn collect-execution-trace-for-test [{:keys [test-class output-dir], :as arg}]
  (let [outfile-name (get-test-name test-class)]
    (run-execution-trace-collector (assoc arg :output (join-path output-dir outfile-name)))))

(defn collect-execution-trace-for-project [{:keys [project-root test-folder target-folder agent-jar output-dir]}]
  (let [test-folder-abs (join-path project-root test-folder)
        target-folder-abs (join-path project-root target-folder)
        classpath (str test-folder-abs ":" target-folder-abs ":" (gen-classpath project-root))
        test-classes (get-test-classes (classes-in-dir test-folder-abs))
        classes-to-instrument (str/join ":" (classes-in-dir target-folder-abs))
        arg {:classes-to-instrument classes-to-instrument
             :agent-jar agent-jar
             :classpath classpath
             :output-dir output-dir}]
    (doall (map #(collect-execution-trace-for-test (assoc arg :test-class %)) (sort test-classes)))))

(s/def ::project-root string?)
(s/def ::test-folder string?)
(s/def ::target-folder string?)
(s/def ::agent-jar string?)
(s/def ::output-dir string?)
(s/def ::project-description (s/keys :req [::project-root ::test-folder ::target-folder ::agent-jar ::output-dir]))

(s/fdef collect-execution-trace-for-project
  :args ::project-description)