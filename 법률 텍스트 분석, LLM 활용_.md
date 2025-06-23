

# **상징적 비계에서 신경-상징적 시너지까지: LLM 시대의 법률 텍스트 분석 재평가**

---

## **제1장: LLM 이전 시대의 법률 텍스트 분석: Ashley (2017) 리뷰**

본 섹션에서는 Ashley의 저서 1 6장부터 10장까지 기술된 방법론을 상세히 분석하여, 해당 시대의 기본 개념과 주요 기술적 과제를 정립한다. 이 시대의 핵심 주제는 "지식 획득 병목 현상(knowledge acquisition bottleneck)"으로, 당시 시스템이 의존했던 정형화된 지식 표현을 구축하는 데 막대한 인적 노력이 필요했다는 점이다.

### **1.1. 법률 지식의 공학: 온톨로지, 타입 시스템, 그리고 구조를 향한 탐구 (6장)**

#### **핵심 개념**

이 소항목에서는 법률 지식을 표현하는 데 사용된 기본적인 상징적 구조를 요약한다. 법률 온톨로지는 특정 법률 영역 내의 개념과 그 관계(예: is-a, has-as-parts)를 명시적이고 형식적으로 정의하는 명세서로서의 목적을 갖는다.1 당시에는 전문가 지식과 초기 자연어 처리(NLP)/기계 학습(ML) 기술을 결합하여 말뭉치에서 용어를 추출하는 하이브리드 상향식/하향식 접근법의 대표적 사례인 DALOS 온톨로지를 포함하여, 수동 및 반자동 구축 방법이 널리 사용되었다.1

#### **UIMA와 LUIMA 타입 시스템**

핵심적인 초점은 비정형 정보 관리 아키텍처(UIMA) 프레임워크와 이를 법률 분야에 적용한 LUIMA에 맞춰질 것이다. 타입 시스템은 텍스트 분석을 위한 특화된 온톨로지로서, 의미론적 마크업 구조를 정의하고 파이프라인 내의 소프트웨어 구성요소("애너테이터") 간의 통신을 조정한다.1 LUIMA 타입 시스템의 계층적 구조(용어, 언급, 표현, 문장 수준 타입)를 상세히 설명하며, 이것이 사법 판결문에서 문장의 논증적 역할(예:

LegalRuleSentence, EvidenceBasedFindingSentence)을 포착하도록 설계되었음을 밝힐 것이다.1 그 목표는 단순한 키워드 매칭을 넘어 보다 정밀한 개념적 정보 검색을 가능하게 하는 것이었다.1

이러한 방법론은 의미론보다 구조를 우선시하는 패러다임을 드러낸다. 6장에서 기술된 방법들은 일차적 과제가 인간이 설계한 엄격한 논리적 구조(온톨로지나 타입 시스템)를 비정형 텍스트에 부과하는 것이었음을 보여준다. 시스템의 "이해"는 이 구조에 명시적으로 정의된 개념과 관계에 국한되었다. 예를 들어, DALOS 프로젝트는 통계적 방법을 사용하여 후보 용어(fuzzynyms)를 찾았지만, 실제 의미를 부여하고 이를 형식적 온톨로지에 연결하는 것은 여전히 인간 전문가의 몫이었다.1 이는 인간의 논리가 구조를 지시하고, 기계의 역할은 그 미리 정의된 구조에 맞는 사례를 찾는 작업 흐름을 보여준다. 이는 데이터 자체에서 창발적(emergent) 의미 관계를 도출하는 현대의 LLM과는 극명한 대조를 이룬다. 사전 정의된 구조에 대한 이러한 과도한 의존은 특정 영역 내에서는 강력하지만, 새로운 법률 영역으로 확장하거나 적응하기에는 취약하고 엄청난 비용이 드는 시스템을 만들어냈다. LUIMA 타입 시스템은 백신 상해 사건에서는 강력했지만, 예를 들어 계약법에 적용하기 위해서는 새롭고 집중적인 개발 주기가 필요했을 것이다.1

### **1.2. 초기 AI 강화 법률 검색의 메커니즘 (7장)**

#### **핵심 개념**

이 소항목에서는 법률 정보 검색(IR)이 불리언 및 벡터 공간 모델(TF-IDF 사용)에서 베이지안 네트워크와 같은 더 정교한 확률 모델로 진화한 과정을 요약한다.1 여기서 확인된 핵심 한계는 이러한 시스템이 효율적이기는 하지만, "어휘적 격차(lexical gap)"를 극복하지 못하고 문서의 심층적인 법적 관련성을 포착할 수 없다는 점이었다.

#### **AI 및 법률 분야의 개선 방안**

IR을 "더 스마트하게" 만들기 위한 두 가지 주요 AI 및 법률 접근법을 상세히 설명할 것이다:

* **SPIRE:** 이 시스템은 사례 기반 추론기(HYPO/CATO 등)를 전문(full-text) IR 시스템과 통합했다. 수동으로 정의된 법률 "요소(factor)"를 사용하여 검색을 시작하고, 초기 사례 집합을 검색한 다음, 해당 사례들의 전문을 "관련성 피드백(relevance feedback)"으로 사용하여 더 큰 말뭉치에서 더 유사한 사례를 찾는 방식이었다. 이는 정형화된 법률 지식을 IR 프로세스에 직접 주입하려는 시도였다.1  
* **SCALIR:** 이 프로젝트는 사례, 법규, 용어의 연결망 네트워크를 사용했다. 관련성은 네트워크를 통해 "활성화 전파(spreading activation)"로 결정되었으며, 이를 통해 인용 링크나 다른 관계를 따라가며 용어가 명시적으로 존재하지 않더라도 개념적 검색이 가능했다.1

SPIRE나 SCALIR과 같은 시스템들은 본질적으로 두 개의 분리된 세계, 즉 정형화된 법률 지식의 상징적 세계(요소, 규칙)와 전문 IR의 통계적 세계를 연결하는 "다리"였다. 이들은 두 세계를 통합하지 못했다. SPIRE는 두 개의 별도 데이터베이스(요소화된 사례용, 전문 구절용)를 유지하고 그 사이에서 정보를 주고받아야 했다.1 이러한 아키텍처의 복잡성은 단일 모델이 정형화된 추론과 자연어를 모두 처리할 수 없었던 데서 비롯된 직접적인 결과였다. 이는 해당 시대의 근본적인 설계 제약을 보여준다. 목표는 통계적 방법에 상징적 지식을 보강하는 것이었지만, 그 통합은 피상적이었다. 이는 신경망과 상징적 구성요소가 단일하고 일관된 아키텍처 내에서 깊이 통합되는 현대의 신경-상징적 접근 방식과 대조된다. 2017년의 시스템들은 하이브리드였지만, 미래는 진정한 시너지에 있다.

### **1.3. 지도 학습과 논증 마이닝: 데이터 기반 법률 분석의 서막 (8, 9, 10장)**

#### **핵심 개념**

이 소항목에서는 지도 학습(supervised ML)을 법률 텍스트에 적용한 사례를 종합적으로 다룬다.

* **전자증거개시(e-Discovery)에서의 예측 코딩 (8장):** 방대하고 이질적인 ESI에서 관련 문서를 식별하기 위해 ML 분류기를 사용하는 방법을 다룬다. 인간의 피드백을 통한 반복적 훈련 과정이 상세히 설명될 것이다.1  
* **서포트 벡터 머신 (SVMs):** 이 시대의 핵심 알고리즘으로서 SVM을 설명하며, 고차원 특징 공간에서 데이터를 분리하는 최적의 초평면(hyperplane)을 찾는 능력에 초점을 맞춘다. Westlaw History Project가 주요 사례로 사용될 것이며, 여기서 SVM은 "제목 유사성(Title Similarity)" 및 "사건 번호 일치(Docket Match)"와 같은 특징에 대해 훈련되어 항소심 체인에서 이전 사례를 높은 재현율(recall)과 정밀도(precision)로 식별했다.1  
* **정보 추출 (9장 & 10장):** 법규와 판례에서 정형화된 정보를 추출하려는 노력을 상세히 다룬다. 여기에는 법규에서 기능적 정보를 추출하기 위한 지식 공학(KE) 대 ML 논쟁 1과 판례법에서 요소(SMILE+IBP) 및 문장 역할(LUIMA-Annotate)과 같은 논증 관련 정보를 추출하는 내용이 포함된다.1 SMILE+IBP 시스템은 완전한 텍스트 기반 사례 기반 추론(CBR) 파이프라인의 중요한 예시이다. SMILE은 ML을 사용하여 텍스트에서 요소를 추출하고, 이를 IBP 상징적 추론기에 입력하여 결과를 예측했다.1

법률 분야에서의 성능의 역설은 주목할 만하다. NLP의 일반적인 추세는 트랜스포머 기반 LLM이 SVM과 같은 구형 모델을 압도적으로 능가하는 것이지만, 최근 연구에 따르면 법률 분야에서는 이러한 성능 격차가 현저히 작다.16 일부 법률 텍스트 분류 벤치마크에서는 잘 최적화된 SVM이 범용 BERT 모델과 경쟁하거나 심지어 더 나은 성능을 보일 수 있다.16 이는 고도로 정형화되고, 형식적이며, 용어 특화적인 법률 언어의 특성이 SVM의 특징 기반 접근 방식에 특히 적합하다는 것을 시사한다. 일반 언어에서는 약점인 "단어 뭉치(bag-of-words)" 가정이 특정 법률 전문 용어가 막대한 가중치를 가질 때는 덜 치명적이다. 이 발견은 매우 중요하다. 이는 LLM 이전 시대에 개발된 상징적이고 특징이 풍부한 표현(Ashley의 요소 등)이 구식이 아니라는 것을 의미한다. 그것들은 LLM이 의미론적 유창성에도 불구하고 명시적인 지침 없이는 완전히 활용하지 못할 수 있는 구조적 지식의 한 유형을 포착한다. 이는 상징적 특징이 LLM의 생성 능력을 구조화하고 제약하는 데 사용될 수 있는 하이브리드 신경-상징 시스템에 대한 주장을 강화한다.

* **표 1: 법률 분석 패러다임 비교 분석**

| 비교 차원 | LLM 이전 패러다임 (상징적/지도학습) | 현대 LLM 패러다임 (생성형) |
| :---- | :---- | :---- |
| **지식 표현** | 수작업 온톨로지, 타입 시스템, 요소 계층 | 학습된 분산 표현 (임베딩) |
| **주요 추론 방식** | 규칙 기반 연역, 사례 기반 유추 추론 | 확률적 다음 토큰 예측, 패턴 완성 |
| **설명가능성** | 높음 (규칙/요소로 추적 가능) | 낮음 (불투명한 "블랙박스") |
| **확장성** | 낮음 (수작업 지식 공학에 의해 제한됨) | 높음 (데이터와 컴퓨팅 파워에 따라 확장) |
| **인간의 역할** | 지식 공학자, 애너테이터 | 프롬프트 엔지니어, 검증자 |
| **핵심 과제** | 지식 획득 병목 현상 | 환각(Hallucination) 및 신뢰성 |

---

## **제2장: 패러다임 전환: 법률 분야에서 대규모 언어 모델의 역량과 한계**

본 섹션에서는 2017년부터 현재까지의 기술적 격차를 해소하고, LLM이 법률 분석 분야에 미친 파괴적인 영향을 비판적으로 평가한다.

### **2.1. 지식 획득 병목 현상의 종말?**

#### **자동화된 지식 추출**

LLM은 지식 표현의 경제학을 근본적으로 변화시킨다. LLM은 이전에 광범위한 수동 주석 작업과 규칙 작성이 필요했던 바로 그 작업들을 제로샷(zero-shot) 또는 퓨샷(few-shot) 방식으로 프롬프트를 통해 수행할 수 있다.18

* **온톨로지/지식 그래프 구축:** LLM은 비정형 법률 텍스트를 파싱하여 개체(entity)와 관계(relationship)를 자동으로 추출하고, 최소한의 인간 개입으로 지식 그래프를 채울 수 있다.24 이는 Ashley의 6장에서 설명된 작업을 직접적으로 자동화한다.  
* **논증 마이닝:** LLM은 텍스트에서 논증 구성 요소(주장, 전제)와 관계(지지, 공격)를 식별하는 데 강력한 성능을 보여주었으며, 이는 Ashley의 10장의 핵심 초점이었던 작업이다.30

이러한 변화는 작업 흐름의 역전을 의미한다. LLM 이전 시대의 작업 흐름은 인간 전문가 → 상징적 모델 → 데이터였다. 전문가가 모델(온톨로지, 규칙)을 만들면 기계는 그 모델에 맞는 데이터를 찾는 역할을 했다. LLM 시대에는 이 흐름이 데이터 → LLM (신경망 모델) → 상징적 모델로 역전된다. 기계가 원시 데이터를 처리하여 상징적 모델을 *생성*하면, 인간 전문가는 이를 검증하는 역할을 맡는다. 이러한 역전은 이전에는 상상할 수 없었던 대규모의 법률 분야별 지식 기반을 구축하는 것을 실현 가능하게 만든다. 병목 현상이 사라진 것은 아니지만, *창조*에서 *검증 및 큐레이션*으로 이동했으며, 이는 훨씬 더 다루기 쉬운 문제이다.

### **2.2. "블랙박스" 딜레마: 환각, 불투명성, 그리고 논리적 취약성**

#### **신뢰 격차**

강력한 성능에도 불구하고, 독립적으로 사용되는 LLM은 본질적인 한계로 인해 고위험 법률 응용 분야에는 근본적으로 부적합하다.

* **환각(Hallucination):** 환각은 사실적 정확성과 검증 가능한 출처가 가장 중요한 법률 분야에서 치명적인 실패 모드이다.24  
* **설명가능성(XAI)의 부재:** LLM은 "블랙박스" 모델이다. 출력에 대한 단계별 논리적 정당성을 제공할 수 없으며, 이는 법적 추론과 적법 절차에 있어 타협할 수 없는 요구 사항이다.37  
* **논리적 취약성:** 통계적 패턴에 기반한 생성 모델로서, LLM은 법규 해석 및 규칙 적용에 필요한 엄격하고 형식적인 논리를 다루는 데 어려움을 겪는다.35

"확률적 앵무새(stochastic parrot)"라는 용어는 LLM의 작동 방식을 잘 포착한다. 즉, 언어적 패턴을 능숙하게 모방하지만, 세계에 대한 진정한 모델이나 논리적 연역 능력은 부족하다. Ashley가 설명한 법적 추론은 단순히 텍스트에서 패턴을 찾는 것이 아니라, 규칙을 적용하고, 구조화된 비교를 통해 유추하며, 논리적 주장을 구성하는 것이다.1 독립적으로 사용되는 LLM은 법적 추론을 수행할 수 없다. 법적 추론처럼

*보이는* 텍스트를 생성할 수는 있지만, 신뢰할 수는 없다. 이는 LLM의 출력을 형식적이고 상징적인 시스템을 사용하여 제약하고 검증할 수 있는 하이브리드 아키텍처의 절대적인 필요성을 확립한다.

---

## **제3장: 미래를 위한 제안: 법률 지능을 위한 신경-상징 아키텍처**

본 섹션은 이 보고서의 핵심 논지인, LLM 이전 패러다임과 현대 패러다임의 장점을 결합한 구체적이고 통합된 아키텍처를 신경-상징 시스템으로 명시적으로 제안한다.

### **3.1. 신경-상징적 전제: 인식과 추론의 융합**

신경-상징 AI는 신경망의 인식 능력(비정형 데이터 처리, 의미론 이해)과 상징 논리의 추론 능력(규칙, 그래프, 설명가능성)을 결합한 하이브리드 접근법이다.43 이 패러다임은 법률 분야의 이중적 요구, 즉 자연어 텍스트의 미묘한 차이를 이해하면서도 형식적인 법률 규칙의 엄격함을 준수해야 하는 필요성을 직접적으로 해결한다. 우리는 이 아키텍처를 설명하기 위해 "시스템 1"(빠르고 직관적인 신경망)과 "시스템 2"(느리고 신중한 상징적) 인지 비유를 사용할 것이다.48

### **3.2. 1단계 (신경망 → 상징): LLM을 이용한 법률 지식 그래프 구축**

#### **방법론**

이 소항목은 신경-상징 루프의 전반부, 즉 신경망 구성요소(LLM)를 사용하여 상징적 구성요소(지식 그래프, KG)를 만드는 과정을 상세히 설명한다.

1. **스키마 정의 (인간 참여):** 법률 전문가들이 관심 법률 분야(예: 계약법, 불법행위법)에 대한 견고한 온톨로지 스키마를 정의한다. 이 스키마는 Ashley의 연구에서 제시된 원칙에 기반하여 조항, 의무, 권리, 판례, 요소, 판결과 같은 개체와 그들 간의 관계를 정의할 것이다.1  
2. **데이터 수집:** 비정형 문서 말뭉치(예: 수천 건의 상업 계약서, 판례 모음)를 수집한다.25  
3. **LLM 기반 정보 추출:** LLM에게 각 문서를 읽고 스키마에 정의된 개체와 관계의 인스턴스를 추출하도록 프롬프트를 제공한다. 예를 들어, "책임 제한" 조항을 식별하고, 당사자 및 책임 한도를 추출하여 이를 구조화된 삼중항(triples) 집합으로 표현한다.28  
4. **KG 구축 및 벡터화:** 추출된 삼중항은 그래프 데이터베이스를 채운다. 동시에 정보가 추출된 텍스트 청크는 벡터 임베딩으로 변환되어 저장되며, 이는 상징적 노드를 의미론적 표현과 연결한다.25

이 과정은 구조화되고, 감사 가능하며, 도메인 특화적인 지식 기반을 생성한다. LLM이 훈련받는 비차별적인 데이터와 달리, 이 KG의 모든 정보는 출처 텍스트에 명시적으로 연결된다. 이 KG는 시스템의 "장기 기억"이자 단일 진실 공급원(single source of truth)이 된다. 이는 신경망 구성요소가 추론의 근거로 삼아야 할 세계에 대한 상징적 표현이며, 사실을 날조하는 것을 방지한다.

### **3.3. 2단계 (상징 → 신경망): 근거 기반의 설명 가능한 추론을 위한 Graph-RAG**

#### **방법론**

이 소항목은 루프의 후반부, 즉 상징적 KG를 사용하여 신경망 LLM을 안내하고 근거를 제시하는 과정을 상세히 설명한다.

1. **자연어 질의:** 사용자가 복잡한 법률 질문을 한다. 예: "2022년 이후 체결된 EU 기반 고객과의 계약에서 불가항력(force majeure) 조항에 대한 우리의 표준 책임 한도는 얼마입니까?".50  
2. **의미론적 \+ 구조적 검색:** 질의는 하이브리드 검색 프로세스에 사용된다. 의미론적 의미는 벡터 검색을 통해 KG에서 관련 노드를 찾는 데 사용된다. 질의의 구조화된 부분("EU 기반 고객", "2022년 이후")은 이러한 정확한 논리적 제약 조건과 일치하는 노드를 찾기 위해 KG를 탐색하는 형식적 그래프 쿼리(예: Cypher, SPARQL)로 변환된다.36  
3. **컨텍스트 증강:** 시스템은 KG에서 관련성 있고 검증 가능한 사실들의 정확한 하위 그래프(예: 특정 계약, 조항, 책임 한도 목록)를 검색한다. 이 하위 그래프가 "검색된 컨텍스트"가 된다.  
4. **인용을 포함한 LLM 생성:** LLM에게 검색된 컨텍스트와 원본 질문이 주어지며, "제공된 정보만을 사용하여 사용자의 질문에 답하고, 모든 진술에 대한 출처를 인용하라"는 엄격한 지침이 함께 제공된다. 이제 LLM의 역할은 "지식 보유자"에서 "유창한 종합자 및 설명자"로 축소된다.50 출력은 KG의 노드로 직접 연결되는 인용이 포함된 자연어 답변이다.

이 검색-증강 생성(RAG) 아키텍처, 특히 Graph-RAG는 종단 간 설명가능성을 제공한다. 사용자는 최종 답변의 인용을 클릭하여 KG의 특정 노드/관계로 이동할 수 있으며, 이는 다시 사실이 추출된 원본 문서의 정확한 문장으로 연결된다. 이는 "블랙박스" 문제를 해결한다. 추론 과정은 더 이상 신경망의 가중치 안에 숨겨져 있지 않고, 검색 및 종합 단계에서 명시적으로 드러난다. 이는 투명성과 책임성에 대한 법률적 요구 사항과 직접적으로 일치한다.38

* **표 2: 신경-상징적 법률 RAG 시스템의 아키텍처**

| 단계 | 구성요소/과정 | 입력 | 처리 | 출력 |
| :---- | :---- | :---- | :---- | :---- |
| **1\. 오프라인** | **KG 구축 (신경망 → 상징)** | 비정형 법률 말뭉치 (판례, 법규, 계약서) | 인간 정의 스키마에 기반한 LLM 기반 NER 및 RE | 채워진 법률 지식 그래프 (상징적) \+ 벡터 인덱스 (신경망) |
| **2\. 온라인** | **질의 응답 (상징 → 신경망)** | 사용자의 자연어 질의 | **1 (검색):** KG에 대한 하이브리드 질의 (구조적 질의 \+ 벡터 검색) | **1:** 검색된 하위 그래프 (검증 가능한 컨텍스트) |
|  |  |  | **2 (생성):** 검색된 컨텍스트만을 기반으로 LLM이 답변 종합 | **2:** 추적 가능한 인용이 포함된 자연어 답변 |

---

## **제4장: 결론: 신뢰할 수 있고 확장 가능한 법률 AI를 향하여**

본 보고서의 논지를 종합하면, Ashley 1가 기술한 상징적 접근법은 구조화되고 설명 가능한 추론의 필요성을 정확하게 파악했지만, "지식 획득 병목 현상"이라는 한계에 부딪혔다. 현대의 LLM은 강력하지만, 새로운 "신뢰성 병목 현상"을 야기했다. 제안된 신경-상징 아키텍처는 이 두 가지 병목 현상을 동시에 해결한다. RAG 프레임워크 내에서 LLM을 사용하여 상징적 지식 그래프를 구축하고 질의함으로써, 우리는 확장 가능하고, 의미를 인식하며, 논리적으로 엄격하고, 완전히 설명 가능한 법률 AI 시스템을 만들 수 있다. 이는 마침내 법률 분야에서 AI의 오랜 약속을 실현하는 길이다.

#### **참고 자료**

1. Artificial Intelligence and Legal Analytics \- Kevin D. Ashley.pdf  
2. Applied Legal Analytics & AI | The LUIMA Group, 6월 23, 2025에 액세스, [https://luimagroup.github.io/appliedlegalanalytics/](https://luimagroup.github.io/appliedlegalanalytics/)  
3. Kevin D. Ashley | School of Law, 6월 23, 2025에 액세스, [https://www.law.pitt.edu/people/kevin-d-ashley](https://www.law.pitt.edu/people/kevin-d-ashley)  
4. Text Categorization with Support Vector Machines: Learning with Many Relevant Features \- CS@Cornell, 6월 23, 2025에 액세스, [https://www.cs.cornell.edu/\~tj/publications/joachims\_98a.pdf](https://www.cs.cornell.edu/~tj/publications/joachims_98a.pdf)  
5. Comparing Support Vector Machines and Decision Trees for Text Classification, 6월 23, 2025에 액세스, [https://www.geeksforgeeks.org/comparing-support-vector-machines-and-decision-trees-for-text-classification/](https://www.geeksforgeeks.org/comparing-support-vector-machines-and-decision-trees-for-text-classification/)  
6. A Statistical Learning Model of Text Classification with Support Vector Machines \- CS@Cornell, 6월 23, 2025에 액세스, [https://www.cs.cornell.edu/\~tj/publications/joachims\_01a.pdf](https://www.cs.cornell.edu/~tj/publications/joachims_01a.pdf)  
7. Information Extraction: Types, Purposes, Best Practices \- Artsyl, 6월 23, 2025에 액세스, [https://www.artsyltech.com/information-extraction](https://www.artsyltech.com/information-extraction)  
8. What is Information Extraction? \- IBM, 6월 23, 2025에 액세스, [https://www.ibm.com/think/topics/information-extraction](https://www.ibm.com/think/topics/information-extraction)  
9. RULE-BASED INFORMATION EXTRACTION: ADVANTAGES, LIMITATIONS, AND PERSPECTIVES, 6월 23, 2025에 액세스, [https://wwwmatthes.in.tum.de/file/47fs4e04rvtp/Sebis-Public-Website/-/Rule-based-Information-Extraction-Advantages-Limitations-and-Perspectives/Wa18b.pdf](https://wwwmatthes.in.tum.de/file/47fs4e04rvtp/Sebis-Public-Website/-/Rule-based-Information-Extraction-Advantages-Limitations-and-Perspectives/Wa18b.pdf)  
10. What is the difference between rule-based and statistical modeling in natural language processing systems? \- Reddit, 6월 23, 2025에 액세스, [https://www.reddit.com/r/LanguageTechnology/comments/64fqap/what\_is\_the\_difference\_between\_rulebased\_and/](https://www.reddit.com/r/LanguageTechnology/comments/64fqap/what_is_the_difference_between_rulebased_and/)  
11. Progress in Textual Case-Based Reasoning: Predicting the Outcome of Legal Cases from Text \- ResearchGate, 6월 23, 2025에 액세스, [https://www.researchgate.net/profile/Kevin-Ashley/publication/221603005\_Progress\_in\_Textual\_Case-Based\_Reasoning\_Predicting\_the\_Outcome\_of\_Legal\_Cases\_from\_Text/links/597141c50f7e9b25e8606094/Progress-in-Textual-Case-Based-Reasoning-Predicting-the-Outcome-of-Legal-Cases-from-Text.pdf?origin=scientificContributions](https://www.researchgate.net/profile/Kevin-Ashley/publication/221603005_Progress_in_Textual_Case-Based_Reasoning_Predicting_the_Outcome_of_Legal_Cases_from_Text/links/597141c50f7e9b25e8606094/Progress-in-Textual-Case-Based-Reasoning-Predicting-the-Outcome-of-Legal-Cases-from-Text.pdf?origin=scientificContributions)  
12. Progress in Textual Case-Based Reasoning: Predicting the Outcome of Legal Cases from Text. \- ResearchGate, 6월 23, 2025에 액세스, [https://www.researchgate.net/publication/221603005\_Progress\_in\_Textual\_Case-Based\_Reasoning\_Predicting\_the\_Outcome\_of\_Legal\_Cases\_from\_Text](https://www.researchgate.net/publication/221603005_Progress_in_Textual_Case-Based_Reasoning_Predicting_the_Outcome_of_Legal_Cases_from_Text)  
13. Textual case-based reasoning \- CORE, 6월 23, 2025에 액세스, [https://core.ac.uk/download/pdf/190329926.pdf](https://core.ac.uk/download/pdf/190329926.pdf)  
14. Case-based reasoning and law, 6월 23, 2025에 액세스, [https://folk.idi.ntnu.no/agnar/CBR%20papers/KER/14.%20CBR%20and%20law%20(Rissland,%20Ashley,%20&%20Branting).pdf](https://folk.idi.ntnu.no/agnar/CBR%20papers/KER/14.%20CBR%20and%20law%20\(Rissland,%20Ashley,%20&%20Branting\).pdf)  
15. Symposium: Legal Reasoning and Artificial Intelligence: How Computers Think Like Lawyers \- Chicago Unbound, 6월 23, 2025에 액세스, [https://chicagounbound.uchicago.edu/cgi/viewcontent.cgi?article=12469\&context=journal\_articles](https://chicagounbound.uchicago.edu/cgi/viewcontent.cgi?article=12469&context=journal_articles)  
16. The Unreasonable Effectiveness of the Baseline: Discussing SVMs in Legal Text Classification \- IOS Press Ebooks, 6월 23, 2025에 액세스, [https://ebooks.iospress.nl/pdf/doi/10.3233/FAIA210317](https://ebooks.iospress.nl/pdf/doi/10.3233/FAIA210317)  
17. \[2109.07234\] The Unreasonable Effectiveness of the Baseline: Discussing SVMs in Legal Text Classification \- ar5iv, 6월 23, 2025에 액세스, [https://ar5iv.labs.arxiv.org/html/2109.07234](https://ar5iv.labs.arxiv.org/html/2109.07234)  
18. Zero-shot Topical Text Classification with LLMs \- an Experimental Study \- ACL Anthology, 6월 23, 2025에 액세스, [https://aclanthology.org/2023.findings-emnlp.647.pdf](https://aclanthology.org/2023.findings-emnlp.647.pdf)  
19. The unreasonable effectiveness of large language models in zero-shot semantic annotation of legal texts \- Frontiers, 6월 23, 2025에 액세스, [https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1279794/full](https://www.frontiersin.org/journals/artificial-intelligence/articles/10.3389/frai.2023.1279794/full)  
20. The unreasonable effectiveness of large language models in zero-shot semantic annotation of legal texts \- PMC, 6월 23, 2025에 액세스, [https://pmc.ncbi.nlm.nih.gov/articles/PMC10690809/](https://pmc.ncbi.nlm.nih.gov/articles/PMC10690809/)  
21. Zero-shot text classification with Amazon SageMaker JumpStart | Artificial Intelligence and Machine Learning \- AWS, 6월 23, 2025에 액세스, [https://aws.amazon.com/blogs/machine-learning/zero-shot-text-classification-with-amazon-sagemaker-jumpstart/](https://aws.amazon.com/blogs/machine-learning/zero-shot-text-classification-with-amazon-sagemaker-jumpstart/)  
22. A Comparative Study of Prompting Strategies for Legal Text Classification \- ACL Anthology, 6월 23, 2025에 액세스, [https://aclanthology.org/2023.nllp-1.25.pdf](https://aclanthology.org/2023.nllp-1.25.pdf)  
23. Fine-Tuned 'Small' LLMs (Still) Significantly Outperform Zero-Shot Generative AI Models in Text Classification \- arXiv, 6월 23, 2025에 액세스, [https://arxiv.org/html/2406.08660v1](https://arxiv.org/html/2406.08660v1)  
24. Knowledge Graphs and LLMs: How Can We Move from Structured Knowledge to AI-Generated Answers? – Part II. \- Constitutional Discourse, 6월 23, 2025에 액세스, [https://constitutionaldiscourse.com/knowledge-graphs-and-llms-how-can-we-move-from-structured-knowledge-to-ai-generated-answers-part-ii/](https://constitutionaldiscourse.com/knowledge-graphs-and-llms-how-can-we-move-from-structured-knowledge-to-ai-generated-answers-part-ii/)  
25. LLM Knowledge Graph Builder: From Zero to GraphRAG in Five Minutes \- Neo4j, 6월 23, 2025에 액세스, [https://neo4j.com/blog/developer/graphrag-llm-knowledge-graph-builder/](https://neo4j.com/blog/developer/graphrag-llm-knowledge-graph-builder/)  
26. Knowledge Graph-Based Legal Query System with LLM and Retrieval Augmented Generation | Request PDF \- ResearchGate, 6월 23, 2025에 액세스, [https://www.researchgate.net/publication/391184550\_Knowledge\_Graph-Based\_Legal\_Query\_System\_with\_LLM\_and\_Retrieval\_Augmented\_Generation](https://www.researchgate.net/publication/391184550_Knowledge_Graph-Based_Legal_Query_System_with_LLM_and_Retrieval_Augmented_Generation)  
27. Leverage Knowledge Graph and Large Language Model for Law Article Recommendation: A Case Study of Chinese Criminal Law \- arXiv, 6월 23, 2025에 액세스, [https://arxiv.org/html/2410.04949v2](https://arxiv.org/html/2410.04949v2)  
28. Construction of Legal Knowledge Graph Based on Knowledge-Enhanced Large Language Models \- ResearchGate, 6월 23, 2025에 액세스, [https://www.researchgate.net/publication/385191261\_Construction\_of\_Legal\_Knowledge\_Graph\_Based\_on\_Knowledge-Enhanced\_Large\_Language\_Models](https://www.researchgate.net/publication/385191261_Construction_of_Legal_Knowledge_Graph_Based_on_Knowledge-Enhanced_Large_Language_Models)  
29. Construction of Legal Knowledge Graph Based on Knowledge-Enhanced Large Language Models \- MDPI, 6월 23, 2025에 액세스, [https://www.mdpi.com/2078-2489/15/11/666](https://www.mdpi.com/2078-2489/15/11/666)  
30. LLMs for Argument Mining: Detection, Extraction, and Relationship Classification of pre-defined Arguments in Online Comments \- arXiv, 6월 23, 2025에 액세스, [https://arxiv.org/html/2505.22956v1](https://arxiv.org/html/2505.22956v1)  
31. Can Large Language Models perform Relation-based Argument Mining? \- ACL Anthology, 6월 23, 2025에 액세스, [https://aclanthology.org/2025.coling-main.569.pdf](https://aclanthology.org/2025.coling-main.569.pdf)  
32. Can Large Language Models perform Relation-based Argument Mining? \- arXiv, 6월 23, 2025에 액세스, [https://arxiv.org/html/2402.11243v1](https://arxiv.org/html/2402.11243v1)  
33. Objection, your honor\!: an LLM-driven approach for generating Korean criminal case counterarguments. \- PhilPapers, 6월 23, 2025에 액세스, [https://philpapers.org/rec/PAROYH](https://philpapers.org/rec/PAROYH)  
34. ArgueMapper Assistant: Interactive Argument Mining Using Generative Language Models, 6월 23, 2025에 액세스, [https://www.dfki.de/fileadmin/user\_upload/import/15453\_Lenz2025ArgueMapperAssistantInteractive.pdf](https://www.dfki.de/fileadmin/user_upload/import/15453_Lenz2025ArgueMapperAssistantInteractive.pdf)  
35. Elevating Legal LLM Responses: Harnessing ... \- ACL Anthology, 6월 23, 2025에 액세스, [https://aclanthology.org/2025.naacl-long.290.pdf](https://aclanthology.org/2025.naacl-long.290.pdf)  
36. Bridging Legal Knowledge and AI: Retrieval-Augmented Generation with Vector Stores, Knowledge Graphs, and Hierarchical Non-negative Matrix Factorization \- arXiv, 6월 23, 2025에 액세스, [https://arxiv.org/html/2502.20364v1](https://arxiv.org/html/2502.20364v1)  
37. XAI: Explainable Artificial Intelligence \- DARPA, 6월 23, 2025에 액세스, [https://www.darpa.mil/research/programs/explainable-artificial-intelligence](https://www.darpa.mil/research/programs/explainable-artificial-intelligence)  
38. What is Explainable AI (XAI)? \- IBM, 6월 23, 2025에 액세스, [https://www.ibm.com/think/topics/explainable-ai](https://www.ibm.com/think/topics/explainable-ai)  
39. What Is Explainable AI (XAI)? \- Palo Alto Networks, 6월 23, 2025에 액세스, [https://www.paloaltonetworks.com/cyberpedia/explainable-ai](https://www.paloaltonetworks.com/cyberpedia/explainable-ai)  
40. Explainable AI in the Legal Domain: Bridging Technology and Jurisprudence \- IndiaAI, 6월 23, 2025에 액세스, [https://indiaai.gov.in/article/explainable-ai-in-the-legal-domain-bridging-technology-and-jurisprudence](https://indiaai.gov.in/article/explainable-ai-in-the-legal-domain-bridging-technology-and-jurisprudence)  
41. Legal Frameworks for XAI Technologies \- The World Conference on Explainable Artificial Intelligence, 6월 23, 2025에 액세스, [https://xaiworldconference.com/2025/legal-frameworks-for-xai-technologies/](https://xaiworldconference.com/2025/legal-frameworks-for-xai-technologies/)  
42. What legal consideration explainable AI raises \- Nupur Jalan, 6월 23, 2025에 액세스, [https://nupurjalan.com/what-legal-consideration-explainable-ai-raises/](https://nupurjalan.com/what-legal-consideration-explainable-ai-raises/)  
43. Towards Robust Legal Reasoning: Harnessing Logical LLMs in Law \- arXiv, 6월 23, 2025에 액세스, [https://arxiv.org/html/2502.17638v1](https://arxiv.org/html/2502.17638v1)  
44. Neuro-Symbolic AI \- Unaligned Newsletter, 6월 23, 2025에 액세스, [https://www.unaligned.io/p/neuro-symbolic-ai](https://www.unaligned.io/p/neuro-symbolic-ai)  
45. How Symbolic AI is Transforming Legal Practice: A Guide for Modern Law Firms \- SmythOS, 6월 23, 2025에 액세스, [https://smythos.com/managers/legal/symbolic-ai-in-law/](https://smythos.com/managers/legal/symbolic-ai-in-law/)  
46. Neuro-Symbolic AI in 2024: A Systematic Review \- arXiv, 6월 23, 2025에 액세스, [https://arxiv.org/pdf/2501.05435](https://arxiv.org/pdf/2501.05435)  
47. AI Reasoning in Deep Learning Era: From Symbolic AI to Neural–Symbolic AI \- MDPI, 6월 23, 2025에 액세스, [https://www.mdpi.com/2227-7390/13/11/1707](https://www.mdpi.com/2227-7390/13/11/1707)  
48. Neuro-Symbolic AI: A New Approach Shaping the Future of Law | Dive\!, 6월 23, 2025에 액세스, [https://go-dive.net/neuro-symbolic-ai-a-new-approach-shaping-the-future-of-law/](https://go-dive.net/neuro-symbolic-ai-a-new-approach-shaping-the-future-of-law/)  
49. Ontologies and Knowledge Graphs \- DataWalk, 6월 23, 2025에 액세스, [https://datawalk.com/ontologies-and-knowledge-graphs-making-enterprise-data-easy-to-understand/](https://datawalk.com/ontologies-and-knowledge-graphs-making-enterprise-data-easy-to-understand/)  
50. How Law Firms Use RAG to Boost Legal Research \- \- Datategy, 6월 23, 2025에 액세스, [https://www.datategy.net/2025/04/14/how-law-firms-use-rag-to-boost-legal-research/](https://www.datategy.net/2025/04/14/how-law-firms-use-rag-to-boost-legal-research/)  
51. Should law firms trust AI for legal research? \- Jurisconsul, 6월 23, 2025에 액세스, [https://www.jurisconsul.com/post/should-law-firms-trust-ai-for-legal-research](https://www.jurisconsul.com/post/should-law-firms-trust-ai-for-legal-research)  
52. Retrieval-Augmented Generation (RAG) in Legal Research \- Lexemo's, 6월 23, 2025에 액세스, [https://e.lexemo.com/uncategorized-en/retrieval-augmented-generation-rag-in-legal-research/](https://e.lexemo.com/uncategorized-en/retrieval-augmented-generation-rag-in-legal-research/)  
53. What roles will RAG and vector search play in AI-assisted law? \- Milvus, 6월 23, 2025에 액세스, [https://milvus.io/ai-quick-reference/what-roles-will-rag-and-vector-search-play-in-aiassisted-law](https://milvus.io/ai-quick-reference/what-roles-will-rag-and-vector-search-play-in-aiassisted-law)