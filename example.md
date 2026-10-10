# type: noul/choice/score

Each answer has a probability per option, the chosen option and a confidence. Three question types: `choice` (pick one of up to 255 options), `noul` (yes/no, returned as a probability) and `score` (a point on a scale you describe). Several independent questions in one request are answered together.

```sh
curl -H 'Content-Type: application/json' -d '{
  "model": "vlang-research",
  "state": "Please refund my order",
  "questions": {
    "refund": {"type":"noul", "instructions":"Does the customer request a refund?"}
  }
}' http://localhost:8001/v1/decisions | jq
```

```sh
curl http://localhost:8001/v1/decisions -H 'content-type: application/json' -d '{
  "model": "vlang-research",
  "state": "Refund request: the customer says the parcel arrived crushed and wants their money back.",
  "questions": {
    "route": {"type": "choice", "instructions": "Which team should handle this?",
              "criteria": {"1": "Refunds and payments", "2": "Damaged or lost parcels", "3": "Account and login problems"}},
    "angry": {"type": "noul", "instructions": "Is the customer angry?"}
  }
}' | jq

curl http://localhost:8001/v1/decisions -H 'content-type: application/json' -d '{
  "model": "vlang-research",
  "state": "Zoë lives in Paris, not Berlin. Her dog is called Café.",
  "questions": {
    "route": {"type": "choice", "instructions": "Where does Zoë live?",
              "criteria": {"a":"Berlin","b":"Rome","c":"Paris","d":"Madrid"}}
  }
}' | jq

curl http://localhost:8001/v1/decisions -H 'content-type: application/json' -d '{
  "model": "vlang-research",
  "state": "The package is still in the warehouse and has not shipped.",
  "questions": {
    "angry": {"type": "noul", "instructions": "Has the package been delivered?"}
  }
}' | jq
```

all in one:

```sh

curl http://localhost:8001/v1/decisions \-H "Content-Type: application/json" \-d '{
    "model": "vlang-research",
    "state": "Help! My payouts have been failing for 3 days.",
    "questions": {
      "is_urgent": {
        "type": "noul",
        "instructions": "Does this message convey urgency?",
        "criteria": {
          "true": "Explicitly time-sensitive",
          "false": "No urgency expressed"
        }
      },
      "department": {
        "type": "choice",
        "instructions": "Which team should handle this?",
        "criteria": {
          "billing": "Payments, invoicing, refunds",
          "technical": "Bugs, outages, integrations",
          "sales": "Pricing, upgrades, new accounts"
        }
      },
      "frustration": {
        "type": "score",
        "instructions": "How frustrated is the customer?",
        "criteria": ["Calm", "Frustrated", "Very angry"]
      }
    }
  }'
```

```json
{
  "model": "vlang-research-0.5.2-92e1455",
  "answers": {
    "is_urgent": {
      "type": "noul",
      "noul": 0.96
    },
    "department": {
      "type": "choice",
      "choice": "billing",
      "probabilities": {
        "technical": 0.11,
        "sales": 0,
        "billing": 0.89
      },
      "confidence": 0.84
    },
    "frustration": {
      "type": "score",
      "score": 1.04,
      "legend": {
        "0": "Calm",
        "1": "Frustrated",
        "2": "Very angry"
      },
      "probabilities": {
        "0": 0,
        "1": 0.96,
        "2": 0.04
      },
      "confidence": 0.93
    }
  },
  "usage": {
    "input_tokens": 427,
    "output_tokens": 73,
    "cost": 0.000017934
  },
  "id": "gen-dec-uuid",
  "provider": "Vlang"
}
```
