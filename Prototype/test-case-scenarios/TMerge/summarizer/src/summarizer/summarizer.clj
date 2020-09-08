(ns summarizer.summarizer)

(def verb-prep-mapping
  {"add" "to"
   "remove" "from"
   "append" "to"})

(defn get-prep-for-verb [verb]
  (if (contains? verb-prep-mapping verb)
    (verb-prep-mapping verb)
    "for"))

(def abbreviation-expansions
  {"init" "initialize"
   "arg" "argument"
   "args" "arguments"})


(defn summarize-getter [{:keys [action object secondary]}]
  (format "Get's %s's %s" (first object) (second object)))

(defn summarize-setter [{:keys [action object secondary]}]
  (format "Sets %s's %s" (first object) (second object)))

(defn summarize-vb-nn [{:keys [action object secondary]}]
  (format "%s %s's %s" action(first object) (second object)))

(defn summarize-vb [{:keys [action object secondary]}]
  (format "%ss %s" action object))

(defn summarize-single-variable-declaration [{:keys [action object secondary]}]
  (format "Instantiates a new %s named %s" (first object) (second object)))

(defn summarize [descriptor]
  (condp = (:type descriptor)
    "VB#NN" (summarize-vb-nn descriptor)
    "getter" (summarize-getter descriptor)
    "setter" (summarize-setter descriptor)
    "VB" (summarize-vb descriptor)
    :single-variable-declaration (summarize-single-variable-declaration descriptor)
    nil))