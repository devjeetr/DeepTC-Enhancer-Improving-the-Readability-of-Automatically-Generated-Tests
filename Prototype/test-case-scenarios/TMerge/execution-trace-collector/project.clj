(defproject execution-trace-collector "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.10.1"]
                 [junit/junit "4.12"]
                 [javaparser-wrapper "0.1.0-SNAPSHOT"]
                 [com.taoensso/timbre "4.10.0"]
                 ]
  :plugins [[lein-cljfmt "0.6.6"]
            [lein-kibit "0.1.8"]]
  :repl-options {:init-ns execution-trace-collector.core})
