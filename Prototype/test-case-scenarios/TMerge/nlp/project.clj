(defproject nlp "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.10.1"]
                 [clojure-opennlp "0.5.0"]
                 [org.apache.commons/commons-lang3 "3.9"]
                 [org.clojurenlp/core "3.7.0"]
                 ]
  :repl-options {:init-ns nlp.core})
