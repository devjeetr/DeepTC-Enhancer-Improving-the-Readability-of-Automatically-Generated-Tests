(ns summarizer.helpers
  (:require [clojure.test :refer :all]
            [javaparser-wrapper.core :as j]))

(defn wrapStatement
  ([statement package] (format "
                            package %s;
                            public class Wrappa{
                                  public void somefunc() {
                                    %s
                                  }
                             }
      " package statement))

  ([statement] (format "
                                public class Wrappa{
                                      public void someFunc() {
                                        %s
                                      }
                                 }
          " statement)))

(defn parse-statement
  ([statement package]
   (j/parse-str (wrapStatement statement package)))
  ([statement]
   (j/parse-str (wrapStatement statement))))

(defn construct-ast-and-get-nodes [nodeType & statementAndPackage]
  (let [cu (apply parse-statement statementAndPackage)]
    (j/find-all nodeType cu)))

(defn construct-ast [source]
  (j/parse source))

(defn construct-ast-and-get-node [& params]
  (first (apply construct-ast-and-get-nodes params)))
