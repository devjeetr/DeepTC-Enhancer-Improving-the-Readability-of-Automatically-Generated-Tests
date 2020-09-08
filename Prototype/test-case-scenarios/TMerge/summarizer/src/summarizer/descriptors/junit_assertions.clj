(ns summarizer.descriptors.junit-assertions
  (:require [ast-extractor.accessors :as a]
            [summarizer.common :refer [describe-as-value]]
            [clojure.pprint :as pp]))

(defn get-expected [node]
  (let [arguments (a/get-method-arguments node)]
    (condp = (count arguments)
      4 (second arguments)
      3 (second arguments)
      2 (first arguments)
      (throw "Wrong number of arguments in assertion"))))

(defn get-actual [node]
  (let [arguments (a/get-method-arguments node)]
    (condp = (count arguments)
      4 (nth arguments 2)
      3 (last arguments)
      2 (last arguments)
      1 (first arguments)
      (throw "Wrong number of arguments in assertion"))))

(def describe-actual-as-value (comp describe-as-value get-actual))
(def describe-expected-as-value (comp describe-as-value get-expected))

(defn- describe-assert-array-equals [node]
  "
    assertArrayEquals(byte[] expecteds, byte[] actuals)
    assertArrayEquals(double[] expecteds, double[] actuals, double delta)
    assertArrayEquals(float[] expecteds, float[] actuals, float delta)
    assertArrayEquals(String message, byte[] expecteds, byte[] actuals)
    assertArrayEquals(String message, char[] expecteds, char[] actuals)
    assertArrayEquals(String message, double[] expecteds, double[] actuals, double delta)
    assertArrayEquals(String message, float[] expecteds, float[] actuals, float delta)
  ")

(defn- describe-assert-equals [node]
  "
    assertEquals(double expected, double actual, double delta)
    assertEquals(String message, double expected, double actual, double delta)
    assertEquals(String message, long expected, long actual)
    assertEquals(String message, Object expected, Object actual)
  "
  {:type :assertion
   :assertion :array-equals
   :actual (describe-actual-as-value node)
   :expected (describe-expected-as-value node)})

(defn- describe-assert-false [node]
  {:type :assertion
   :assertion :assert-false
   :actual (describe-actual-as-value node)})

(defn- describe-assert-true [node]
  {:type :assertion
   :assertion :assert-true
   :actual (describe-actual-as-value node)})


(defn- describe-assert-null [node]
  {:type :assertion
   :assertion :assert-null
   :actual (describe-actual-as-value node)})

(defn- describe-assert-not-null [node]
  {:type :assertion
   :assertion :assert-not-null
   :actual (describe-actual-as-value node)})

(defn describe-basic-junit-assertion [node]
  (condp = (a/get-method-name node)
    "assertEquals" (describe-assert-equals node)
    "assertArrayEquals" (describe-assert-array-equals node)
    "assertFalse" (describe-assert-false node)
    "assertTrue" (describe-assert-true node)
    "assertNull" (describe-assert-null node)
    "assertNotNull" (describe-assert-not-null node)
    nil))

(defn describe-junit-assertion [node]
  (condp = (a/get-method-name node)
    "assertThat" nil
    (describe-basic-junit-assertion node)))

(defn- describe-junit-assertion-with-hamcrest [node])
