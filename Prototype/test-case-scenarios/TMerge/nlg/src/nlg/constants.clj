(ns nlg.constants
  (:import (simplenlg.features Tense Gender Form)))

(def tense-mapping
  {:future Tense/FUTURE
   :past Tense/PAST
   :present Tense/PRESENT})

(def gender-mapping
  {:feminine Gender/FEMININE
   :masculine Gender/MASCULINE
   :NEUTER Gender/NEUTER})

(def form-mapping
  {:bare-infinitive Form/BARE_INFINITIVE
   :gerund Form/GERUND
   :imperative Form/IMPERATIVE
   :infinitive Form/INFINITIVE
   :normal Form/NORMAL
   :past-participle Form/PAST_PARTICIPLE
   :present-participle Form/PRESENT_PARTICIPLE})

