(defproject summarizer "0.1.0-SNAPSHOT"
  :description "FIXME: write description"
  :url "http://example.com/FIXME"
  :license {:name "EPL-2.0 OR GPL-2.0-or-later WITH Classpath-exception-2.0"
            :url "https://www.eclipse.org/legal/epl-2.0/"}
  :dependencies [[org.clojure/clojure "1.10.1"]
                 [com.taoensso/timbre "4.10.0"]
                 [javaparser-wrapper "0.1.0-SNAPSHOT"]
                 [ast-extractor "0.1.0-SNAPSHOT"]
                 [nlp "0.1.0-SNAPSHOT"]
                 [com.github.javaparser/javaparser-core "3.15.12"]
                 [nlg "0.1.0-SNAPSHOT"]]



  :plugins [[lein-cljfmt "0.6.6"]
            [lein-kibit "0.1.8"]]

  :repl-options {:init-ns summarizer.core})
