(ns javaparser-wrapper.core
  (:require [clojure.spec.alpha :as s]
            [taoensso.timbre :as timbre])
  (:import (java.io File)
           (com.github.javaparser StaticJavaParser TokenRange Position)
           (com.github.javaparser.ast CompilationUnit)
           (com.github.javaparser.ast.expr Expression)
           (com.github.javaparser.resolution Resolvable UnsolvedSymbolException)
           (com.github.javaparser.ast.body MethodDeclaration)
           (com.github.javaparser.ast.stmt Statement)
           (java.util.stream Collectors)
           (java.util Optional)
           (com.github.javaparser.symbolsolver.resolution.typesolvers CombinedTypeSolver ReflectionTypeSolver JavaParserTypeSolver)
           (com.github.javaparser.symbolsolver.model.resolution TypeSolver)
           (com.github.javaparser.symbolsolver JavaSymbolSolver)))

(defn- create-symbol-solver [^String path]
  (let [reflection-solver (ReflectionTypeSolver. false)
        jp-type-solver (JavaParserTypeSolver. path)
        combined-solver (CombinedTypeSolver. (into-array TypeSolver [reflection-solver jp-type-solver]))]

    (new JavaSymbolSolver combined-solver)))

(defn configure-symbol-solver [^String path]
  (let [symbol-solver (create-symbol-solver path)
        config (StaticJavaParser/getConfiguration)]
    (.setSymbolResolver config symbol-solver)
    (timbre/info "Configured JavaParser SymbolSolver with path" path)))

(defn parse ^CompilationUnit [^File file]
  "StaticJavaParser/parse"
  (try (StaticJavaParser/parse file)
       (catch Exception e
         (timbre/warn "Error while parsing file " (.getName file)))))


(defn parse-str [^String string]
  (StaticJavaParser/parse string))

(defn find-all
  ([^Class node-type ^CompilationUnit cu]
   "cu.findAll(node-type)"
   (.findAll cu node-type))
  ([^Class node-type] (fn [^CompilationUnit cu] (find-all node-type cu))))

;; symbol resolution
(defn resolve-node
  "
    Tries to resolve the given node.
    If not resolved, returns nil
  "
  [^Resolvable resolvable]
  (try
    (.resolve resolvable)
    (catch UnsolvedSymbolException e
      (timbre/warn (.getMessage e)))
    (catch IllegalStateException e
      (timbre/warn "Illegal StateException while trying to resolve node " resolvable))
    (catch RuntimeException e
      (timbre/warn "SymbolSolverFailed: " e))))


(defn resolved->ast-node
  "
    Tries to resolve the given node and return the ast
    node of the resolution.
    Returns nil if given node cannot be resolved
  "
  [^Resolvable resolved]
  (when resolved
    (let [ast-node (.toAst resolved)]
      (when (.isPresent ast-node) (.get ast-node)))))

;; expression helpers
(defn get-name-str [^Expression expr]
  (.getNameAsString expr))

(defn get-name [^Expression expr]
  (.getName expr))

(defn get-meta-model-str [expr]
  (str (.getMetaModel expr)))

(defn get-scope [^Expression expr]
  (let [scope (.getScope expr)]
    (when (.isPresent scope) (.get scope))))

(defn get-type [^Expression expr] (.getType expr))
(defn get-type-str [^Expression expr] (str (get-type expr)))
(defn node-list->seq [nodelist]
  (seq (.collect (.stream nodelist) (Collectors/toList))))

(defn get-left [expr] (.getLeft expr))
(defn get-right [expr] (.getRight expr))
(defn get-operator [expr] (.getOperator expr))

(defn get-method-statements [^MethodDeclaration declaration]
  (let [body (.getBody declaration)]
    (when (.isPresent body)
      (-> body
          (.get)
          (.getStatements)
          (node-list->seq)))))

(defn get-method-annotations [^MethodDeclaration declaration]
  (node-list->seq (.getAnnotations declaration)))

(defn get-optional [^Optional optional]
  (when (.isPresent optional) (.get optional)))

(defn- position->line-col [^Position begin ^Position end]
  {
   :start-line (.line begin)
   :end-line   (.line end)
   :start-col  (.column begin)
   :end-col    (.column end)})

(defn get-token-range [node]
  (some->> (.getTokenRange node)
           (get-optional)
           (.toRange)
           (get-optional)
           ((juxt #(.begin %) #(.end %)),)
           (apply position->line-col,)))

;; statement predicates

(defn expression-statement? [^Statement statement] (.isExpressionStmt statement))
(defn assertion-statement? [^Statement statement] (.isAssertStmt statement))