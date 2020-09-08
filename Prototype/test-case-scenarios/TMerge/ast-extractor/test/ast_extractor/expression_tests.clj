(ns ast-extractor.expression-tests
  (:require [clojure.test :refer :all]
            [ast-extractor.helpers :as helpers]
            [ast-extractor.extractable :refer [extract-ast-data]]
            [ast-extractor.expressions]
            [javaparser-wrapper.core :as j])
  (:import (com.github.javaparser.ast.expr ObjectCreationExpr ArrayAccessExpr ArrayInitializerExpr TypeExpr CastExpr UnaryExpr$Operator ConditionalExpr UnaryExpr NameExpr MethodCallExpr)
           (com.github.javaparser.ast.type Type)))



;
;   Array Initializer Expression
;


(deftest test-literal-array-expr
  (let [expr (helpers/construct-ast-and-get-node ArrayInitializerExpr "var x = {1, 2, 3};")
        result (extract-ast-data expr)]
    (is (= (:asString result) "{ 1, 2, 3 }"))))

(deftest test-array-access
  (let [expr (helpers/construct-ast-and-get-node ArrayAccessExpr "var x = new ArrayList<String>(100); x[22];")
        result (extract-ast-data expr)]
    (is (= "22" (:value (:index result))))
    (is (= "x" (:name (:array result))))))



;
; Cast Expression tests
;


(deftest cast-expr-test-primitive
  (let [cu (helpers/parse-statement "int x = (int) 20.0f;")
        expr (first (j/find-all CastExpr cu))
        x (println (j/get-meta-model-str expr))
        result (extract-ast-data expr)]
    (is (= (:nodeType result) "CastExpr"))))

;(deftest cast-expr-test-reference-type
;  (let [cu (helpers/parse-statement "var x = (List<String>) new ArrayList<String>();")
;        expr (first (j/find-all CastExpr cu ))
;        result (extract-ast-data expr)]))

;
;     Type Expression Tests
; TODO:
; complete this(low priority)
(deftest test-type-expr-primitive-type
  (let [cu (helpers/parse-statement "Array::toList;")
        expr (first (j/find-all TypeExpr cu))
        result (extract-ast-data expr)]))

;
;     Type tests
;
(deftest test-primitive-type-int
  (let [cu (helpers/parse-statement "int x;")
        expr (first (j/find-all Type cu))
        result (extract-ast-data expr)]
    (is (:nodeType result) "Type")
    (is (:type result) "int")))

(deftest test-primitive-type-float
  (let [cu (helpers/parse-statement "float x;")
        expr (first (j/find-all Type cu))
        result (extract-ast-data expr)]
    (is (:nodeType result) "Type")
    (is (:type result) "float")))

(deftest test-primitive-type-double
  (let [cu (helpers/parse-statement "double x;")
        expr (first (j/find-all Type cu))
        result (extract-ast-data expr)]
    (is (:nodeType result) "Type")
    (is (:type result) "double")))

(deftest test-reference-type-string
  (let [cu (helpers/parse-statement "String x;")
        expr (first (j/find-all Type cu))
        result (extract-ast-data expr)]
    (is (:nodeType result) "Type")
    (is (:type result) "String")))

(deftest test-reference-type-arraylist-of-strings
  (let [cu (helpers/parse-statement "ArrayList<String> x;")
        expr (first (j/find-all Type cu))
        result (extract-ast-data expr)]
    (is (:nodeType result) "Type")
    (is (:type result) "ArrayList<String>")))

(deftest test-reference-type-custom-class
  (let [cu (helpers/parse-statement "BasicFile x;")
        expr (first (j/find-all Type cu))
        result (extract-ast-data expr)]
    (is (:nodeType result) "Type")
    (is (:type result) "BasicFile")))


;;          Name Expression Tests


(deftest test-name-expr
  (let [expr (helpers/construct-ast-and-get-node NameExpr "int x = 0; int y = x + 10;")
        result (extract-ast-data expr)]
    (is (= "x" (:name result)))))

;;         Unary Expression tests
(deftest test-unary-expr-postfix
  (let [expr
        (helpers/construct-ast-and-get-node
         UnaryExpr "int i = 0; i++;")
        result (extract-ast-data expr)
        childnode (:expression result)]
    (is (= UnaryExpr$Operator/POSTFIX_INCREMENT (:operator result)))
    (is (= "UnaryExpr" (:nodeType result)))
    (is (= "i" (:name childnode)))))

;;
;;         Conditional Expression tests
(deftest test-conditional-expr
  (let [expr
        (helpers/construct-ast-and-get-node
         ConditionalExpr "int i = 0; i = i > 0 ? 20: 1;")
        result (extract-ast-data expr)
        ifCond (:if result)
        thenExpr (:then result)
        elseExpr (:else result)]
    (is (= "ConditionalExpr" (:nodeType result)))
    (is (= "i > 0 ? 20 : 1" (:asString result)))
    (is (= "20" (:value thenExpr)))
    (is (= "1" (:value elseExpr)))))

;; MethodCallExpr
(deftest testMethodCallExpr
  (let [expr (helpers/construct-ast-and-get-node MethodCallExpr "var y = BasicFile.build();")
        result (extract-ast-data expr)]
    (is (= (:nodeType result) "MethodCallExpr") "nodeType is correct")
    (is (= (:returnType result) "BasicFile") "type is correctly entered")
    (is (= (:resolvable result) true) "correctly marked as not resolvable")
    (is (not (nil? (:resolvedNode result))) "no resolved node")
    (is (= (:parameters result) nil)) "parameters correctly mapped"))
;
;
(deftest testMethodCallExpr2
  (let [expr (helpers/construct-ast-and-get-node MethodCallExpr "int x = BasicFile.something();")
        result (extract-ast-data expr)]
    (is (= (:nodeType result) "MethodCallExpr") "nodeType is correct")
    (is (= (:returnType result) "int") "type is correctly entered")
    (is (= (:resolvable result) true) "correctly marked as resolvable")
    (is (not (nil? (:resolvedNode result))) "no resolved node")
    (is (= (:parameters result) nil)) "parameters correctly mapped"))
