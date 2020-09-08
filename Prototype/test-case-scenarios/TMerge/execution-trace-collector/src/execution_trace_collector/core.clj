(ns execution-trace-collector.core
  (:require [execution-trace-collector.maven-classpath :as mc]
            [execution-trace-collector.collector :as c]
            [clojure.string :as str]
            [clojure.java.io :as io]
            [clojure.pprint :as pp]
            [javaparser-wrapper.core :as j])
  (:import (java.io File)))

(def root "/home/devjeetroy/Research/Testing-Project/repositories/repos/commons-cli-1.4-src")
(def java-agent "/home/devjeetroy/Research/Testing-Project/test-summarizer-and-merger/line-execution-trace/build/libs/shadow.jar")
(def output-dir "/home/devjeetroy/Research/Testing-Project/repositories/temp/commons-cli-exec-trace")
(c/collect-execution-trace-for-project {
                                        :project-root root
                                        :test-folder "target/test-classes"
                                        :target-folder "target/classes"
                                        :agent-jar java-agent
                                        :output-dir output-dir
                                        })
(shutdown-agents)