(ns summarizer.common
  (:require [clojure.string :as str]
            [ast-extractor.accessors :as a]))

(defn format-list-output
  "
    input: [1, 2, 3, 4], separator= ', ', connector=' and '
    output: 1, 2, 3 and 4
  "
  [xs & {:keys [separator connector] :or {separator ", " connector " and "}}]
  (if (< (count xs) 2)
    (or (first xs) "")
    (str/join (or connector " and ")
              [(str/join
                (or separator ", ")
                (drop-last xs))
               (last xs)])))

(defn describe-as-value
  "
    Describes an expression as a value
  "
  [descriptor]
  (condp = (a/get-node-type descriptor)
    "NameExpr" (a/get-str-repr descriptor)
    "LiteralExpr" (a/get-str-repr descriptor)
    "StringLiteralExpr" (a/get-str-repr descriptor)
    "BooleanLiteralExpr" (a/get-str-repr descriptor)
    "IntegerLiteralExpr" (a/get-str-repr descriptor)
    "NullLiteralExpr" (a/get-str-repr descriptor)
    "ClassExpr" (a/get-str-repr descriptor)
    "CastExpr" (describe-as-value (a/get-target descriptor))
    "MethodCallExpr" (a/get-str-repr descriptor)
    "FieldAccessExpr" (str "the field " (:name descriptor) " of " (describe-as-value (:scope descriptor)))
    "EnclosedExpr" (describe-as-value (:inner descriptor))  ;TODO special case for math statements
    "Type" (a/get-str-repr descriptor)
    "ArrayType" "array"
    "ClassOrInterfaceType" (:type-name descriptor)
    "PrimitiveType" (a/get-node-type descriptor)
    "UnaryExpr" (a/get-str-repr descriptor)
    (str "describe-value: " descriptor)))


(def describe-list-as-values (comp format-list-output (partial map describe-as-value)))


