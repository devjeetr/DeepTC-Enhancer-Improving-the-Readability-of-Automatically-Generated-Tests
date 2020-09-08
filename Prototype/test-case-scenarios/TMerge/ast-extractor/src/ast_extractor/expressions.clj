(ns ast-extractor.expressions
  (:require [javaparser-wrapper.core :as j]
            [ast-extractor.extractable :refer [extractable extract-ast-data]]
            [ast-extractor.types]
            [ast-extractor.common :refer [extract-default-fields get-parameters get-arguments] :as c]
            [clojure.spec.alpha :as s])
  (:import (com.github.javaparser.ast.body MethodDeclaration VariableDeclarator)
           (com.github.javaparser.ast.expr CastExpr FieldAccessExpr LiteralExpr StringLiteralExpr CharLiteralExpr LongLiteralExpr DoubleLiteralExpr ArrayInitializerExpr ArrayAccessExpr MarkerAnnotationExpr SingleMemberAnnotationExpr NormalAnnotationExpr NullLiteralExpr BooleanLiteralExpr IntegerLiteralExpr VariableDeclarationExpr EnclosedExpr AssignExpr TypeExpr Name NameExpr ConditionalExpr UnaryExpr ClassExpr MethodCallExpr ObjectCreationExpr BinaryExpr ArrayCreationExpr ThisExpr)
           (java.util.stream Collectors)
           (com.github.javaparser.ast.type Type)))

;; extractors
(extend-type FieldAccessExpr
  extractable
  (extract-ast-data [^FieldAccessExpr expr]
    (extract-default-fields
      expr
      {:scope (extract-ast-data (.getScope expr))})))


(s/def ::scope ::c/base-node)
(s/def ::name-str string?)
(s/def ::field-access-expr (s/merge ::c/base-node (s/keys :req [::scope])))
(s/def ::name-expr ::c/base-node)
;(s/def ::cast-expr (s/merge ::c/base-node (s/keys ))) TODO


(extend-type CastExpr
  extractable
  (extract-ast-data [^CastExpr expr]
    (extract-default-fields
      expr
      {:type   (extract-ast-data (j/get-type expr))
       :target (extract-ast-data (.getExpression expr))})))

;; All literal expressions
(extend-type
  LiteralExpr
  extractable
  (extract-ast-data [expr]
    (extract-default-fields expr
                            {})))
(s/def ::literal-expr ::c/base-node)

(extend-type
  StringLiteralExpr
  extractable
  (extract-ast-data [^LiteralExpr expr]
    (extract-default-fields
      expr
      {})))
(s/def ::string-literal-expr ::c/base-node)

(extend-type
  NullLiteralExpr
  extractable
  (extract-ast-data [^LiteralExpr expr]
    (extract-default-fields
      expr
      {})))
(s/def ::null-literal-expr ::c/base-node)

(extend-type
  BooleanLiteralExpr
  extractable
  (extract-ast-data [^LiteralExpr expr]
    (extract-default-fields
      expr
      {})))
(s/def ::boolean-literal-expr ::c/base-node)

(extend-type
  IntegerLiteralExpr
  extractable
  (extract-ast-data [^LiteralExpr expr]
    (extract-default-fields
      expr
      {})))
(s/def ::integer-literal-expr ::c/base-node)

(extend-type
  DoubleLiteralExpr
  extractable
  (extract-ast-data [^DoubleLiteralExpr expr]
    (extract-default-fields
      expr
      {})))
(s/def ::double-literal-expr ::c/base-node)

(extend-type
  LongLiteralExpr
  extractable
  (extract-ast-data [^LongLiteralExpr expr]
    (extract-default-fields
      expr
      {})))
(s/def ::long-literal-expr ::c/base-node)

(extend-type
  CharLiteralExpr
  extractable
  (extract-ast-data [^CharLiteralExpr expr]
    (extract-default-fields
      expr
      {})))
(s/def ::char-literal-expr ::c/base-node)

(extend-type
  BinaryExpr
  extractable
  (extract-ast-data [^BinaryExpr expr]
    (extract-default-fields
      expr
      {:left (extract-ast-data (j/get-left expr))
       :right (extract-ast-data (j/get-right expr))
       :operator (j/get-operator expr)})))

;; Arrays
(extend-type
  ArrayInitializerExpr
  extractable
  (extract-ast-data [^ArrayInitializerExpr expr]
    ;; TODO:
    ;; add fields below
    (extract-default-fields
      expr
      {})))
(s/def ::array-initializer-expr ::c/base-node)

(extend-type
  ArrayAccessExpr
  extractable
  (extract-ast-data [^ArrayAccessExpr expr]
    (extract-default-fields
      expr
      {:index (extract-ast-data (.getIndex expr))
       :array (extract-ast-data (j/get-name expr))})))
(s/def ::index ::c/base-node)
(s/def ::array ::name-expr)
(s/def ::array-access-expr (s/merge ::c/base-node (s/keys :req [::index ::array])))

(extend-type
  ArrayCreationExpr
  extractable
  (extract-ast-data [^ArrayCreationExpr expr]
    (extract-default-fields
      expr
      {}))) ;TODO

;; Annotations
(extend-type
  NormalAnnotationExpr
  extractable
  (extract-ast-data [^NormalAnnotationExpr expr]
    (extract-default-fields
      expr
      {:annotations (map
                      (fn [x] {:name  (j/get-name-str x)    ;TODO: parse name and value as an expression
                               :value (str (.getValue x))})
                      (j/node-list->seq (.getPairs expr)))})))
(s/def ::member-value-pair (s/keys :req [::name-str ::c/base-node]))
(s/def ::annotations (s/coll-of ::member-value-pair))
(s/def ::normal-annotation-expr (s/merge ::c/base-node (s/keys :req [::annotations])))
(extend-type
  MarkerAnnotationExpr
  extractable
  (extract-ast-data [^MarkerAnnotationExpr expr]
    (extract-default-fields
      expr
      {})))

(extend-type
  SingleMemberAnnotationExpr
  extractable
  (extract-ast-data [^SingleMemberAnnotationExpr expr]
    (extract-default-fields
      expr
      {:name  (j/get-name-str expr)
       :value (extract-ast-data (.getMemberValue expr))}))) ;TODO: parse value as string

;; Variable Declarations
(extend-type
  VariableDeclarator
  extractable
  (extract-ast-data [^VariableDeclarator variableDeclarator]
    (extract-default-fields
      variableDeclarator
      {:name        (j/get-name-str variableDeclarator)
       :type        (extract-ast-data (j/get-type variableDeclarator))
       :initializer (when (.isPresent (.getInitializer variableDeclarator)) (extract-ast-data (.get (.getInitializer variableDeclarator))))})))

(s/def ::name any?)
(s/def ::type #(instance? Type %))
(s/def ::initializer ::c/base-node)
(s/def ::variable-declarator (s/merge ::c/base-node (s/keys :req [::name ::type ::initializer])))

(extend-type
  VariableDeclarationExpr
  extractable
  (extract-ast-data [^VariableDeclarationExpr expr]
    (extract-default-fields
      expr
      {:variables (map extract-ast-data (.getVariables expr))})))
(s/def ::variables (s/coll-of ::c/base-node))
(s/def ::variable-declaration-expr (s/merge ::c/base-node
                                            (s/keys :req [::variables])))
; misc
(extend-type EnclosedExpr
  extractable
  (extract-ast-data [^EnclosedExpr expr]
    (extract-default-fields
      expr
      {:inner (extract-ast-data (.getInner expr))})))

(extend-type AssignExpr
  extractable
  (extract-ast-data [^AssignExpr expr]
    (extract-default-fields
      expr
      {:target   (extract-ast-data (.getTarget expr))
       :value    (extract-ast-data (.getValue expr))
       :operator (.getOperator expr)})))

(extend-type
  ClassExpr
  extractable
  (extract-ast-data [^ClassExpr expr]
    (extract-default-fields
      expr
      ;;TODO
      ;; figure out what to extract
      {})))

(extend-type
  ThisExpr
  extractable
  (extract-ast-data [^ThisExpr expr]
    (extract-default-fields
      expr
      {})))

(extend-type
  UnaryExpr
  extractable
  (extract-ast-data [^UnaryExpr expr]
    (extract-default-fields
      expr
      {:expression (extract-ast-data (.getExpression expr))
       :operator   (.getOperator expr)})))

(extend-type
  ConditionalExpr
  extractable
  (extract-ast-data [^ConditionalExpr expr]
    (extract-default-fields
      expr
      {:if   (.getCondition expr)
       :then (extract-ast-data (.getThenExpr expr))
       :else (extract-ast-data (.getElseExpr expr))})))

(extend-type
  NameExpr
  extractable
  (extract-ast-data [^NameExpr expr]
    (extract-default-fields
      expr
      {})))


(extend-type
  Name
  extractable
  (extract-ast-data [^Name name]
    (extract-default-fields
      name
      {:name (.asString name)})))


(extend-type
  TypeExpr
  extractable
  (extract-ast-data [^TypeExpr expr]
    {:nodeType (str (.getMetaModel expr))
     :node     expr
     :type     (extract-ast-data (.getType expr))}))

(extend-type
  ObjectCreationExpr
  extractable
  (extract-ast-data
    [^ObjectCreationExpr expr]
    (let [arguments (map extract-ast-data (get-arguments expr))
          symbol-resolution (j/resolve-node expr)
          resolvedNode (j/resolved->ast-node symbol-resolution)]
      {:node         expr
       :nodeType     (j/get-meta-model-str expr)
       :type         (.getTypeAsString expr)
       :resolvedNode resolvedNode
       :resolvable   (not (nil? resolvedNode))
       :arguments    arguments
       :parameters   (get-parameters resolvedNode)})))

;; Methods
;; MethodCalls
(extend-type MethodDeclaration
  extractable
  (extract-ast-data [^MethodDeclaration expr]
    (extract-default-fields
      expr
      {:name       (j/get-name-str expr)
       :statements (j/get-method-statements expr)
       :type       (extract-ast-data (j/get-type expr))})))

(extend-type MethodCallExpr
  extractable
  (extract-ast-data [^MethodCallExpr expr]
    (let [arguments (map extract-ast-data (get-arguments expr))
          symbol-resolution (j/resolve-node expr)
          resolved-ast-node (j/resolved->ast-node symbol-resolution)
          scope (some->> (j/get-scope expr)
                         (extract-ast-data))]
      (extract-default-fields
        expr
        {:name          (j/get-name-str expr)
         :return-type   (some->> symbol-resolution
                                 (.getReturnType)
                                 (str))
         ;:resolved-node resolved-ast-node TODO figure out if we need to extract more info from here
         :resolvable    (not (nil? resolved-ast-node))
         :arguments     arguments
         :parameters    (get-parameters resolved-ast-node)
         :scope         scope
         :static?       (some->> symbol-resolution
                                 (.isStatic))}))))


(s/def ::resolved-node any?)                                ; TODO
(s/def ::resolvable boolean?)
(s/def ::arguments any?)                                    ;TODO
(s/def ::parameters any?)                                   ;TODO
(s/def ::static? boolean?)
(s/def ::method-call-expr (s/merge ::c/base-node (s/keys :req [::resolved-node
                                                               ::resolvable
                                                               ::arguments
                                                               ::parameters
                                                               ::static?])))
