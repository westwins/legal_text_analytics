

# **법률 지능을 위한 신경-상징 아키텍처: 한국 민사소송기록 2025-07 사례 연구**

## **서론: 신경망의 인식과 상징 논리의 추론 융합**

법률 인공지능(AI) 분야가 마주한 핵심 과제는 자연어로 작성된 법률 문서의 미묘한 의미적 뉘앙스를 깊이 이해하는 능력과, 형식화된 법규와 판례의 엄격한 논리 구조를 준수하는 능력을 동시에 확보하는 것입니다. 대규모 언어 모델(LLM)은 전자에 있어 전례 없는 발전을 보여주었으나, 사실관계의 환각(hallucination)이나 논리적 비약의 위험성으로 인해 후자의 요구사항을 충족시키는 데에는 본질적인 한계를 드러냅니다. 이러한 간극을 메우기 위해 본 보고서는 신경망의 패턴 인식 능력과 상징 시스템의 구조적 추론 능력을 결합한 하이브리드형 신경-상징(Neuro-Symbolic) 아키텍처를 제안하고, 그 구체적인 작동 방식을 실제 민사소송기록을 통해 증명하고자 합니다.

이 아키텍처는 인지과학의 "시스템 1"(빠르고 직관적인 신경망)과 "시스템 2"(느리고 심의적인 상징 체계) 비유를 통해 설명할 수 있습니다.1 시스템 1에 해당하는 LLM이 비정형 텍스트로부터 법률적 의미를 '인식'하여 추출하면, 시스템 2에 해당하는 지식 그래프(Knowledge Graph, KG)가 이를 구조화하고 검증 가능한 사실의 집합으로 고정시킵니다. 이후 사용자의 질의가 발생하면, 시스템은 먼저 시스템 2의 지식 그래프에서 명확한 사실관계를 검색하고, 이를 바탕으로 시스템 1의 LLM이 논리적 추론을 수행하도록 유도합니다. 이 과정을 통해 LLM의 역할은 '전지적 지식 보유자'에서 '제한된 정보에 기반한 유창한 추론가'로 전환되어, 결과물의 신뢰성과 설명가능성을 극대화합니다.

본 연구의 실증적 기반이 될 자료는 5개의 개별적 법률 분쟁이 복잡하게 얽혀 있는 43페이지 분량의 민사소송기록(사건번호 2025-07)입니다.1 이 기록은 다수의 당사자, 복잡한 계약 관계, 긴밀하게 연결된 시계열적 사건들을 포함하고 있어, 제안된 아키텍처의 효용성을 검증하기 위한 이상적인 시험대로 기능할 것입니다.

---

## **제1부: 1단계 \- 비정형 소송기록으로부터 법률 지식 그래프 구축 (신경망 → 상징)**

이 단계는 신경-상징 아키텍처의 전반부에 해당하며, LLM이라는 신경망 도구를 사용하여 비정형 데이터인 소송기록으로부터 정형화되고 기계가독적인 상징적 지식 기반, 즉 법률 지식 그래프(KG)를 구축하는 과정을 상세히 기술합니다.

### **1.1 민사소송 온톨로지 스키마 정의**

지식 그래프 구축의 첫걸음은 법률 전문가가 개입하여 법률 지식의 구조를 정의하는 온톨로지 스키마를 설계하는 것입니다. 이 스키마는 AI가 탐색하고 이해해야 할 개념(개체)과 그들 사이의 관계를 명시하는 청사진 역할을 합니다. 이는 단순한 데이터 목록이 아니라, 법률 논리가 전개될 수 있는 기본 골격을 제공함으로써 시스템의 추론 능력을 담보하는 핵심적인 과정입니다.

#### **개체 유형 (노드)**

* Person: 개인. 속성: name, residentID, address, role(예: 원고, 피고, 채무자, 채권자).  
* Corporation: 법인. 속성: name, registrationNumber, address, representative.  
* LegalEntity: Person과 Corporation의 상위 클래스.  
* Property: 재산. 하위 유형: RealEstate\_Land(토지), RealEstate\_Building(건물), RealEstate\_Condo(집합건물). 속성: address, legalDescription, area.  
* Contract: 계약. 하위 유형: LoanAgreement(대여금 계약), LeaseAgreement(임대차 계약), SalesAgreement(매매 계약), BusinessTransferAgreement(영업양도 계약). 속성: date, parties, monetaryValue.  
* Debt: 채무. 속성: principalAmount, interestRate, dueDate, creditor, debtor.  
* SecurityInterest: 담보물권. 하위 유형: Mortgage(근저당권), Lien(유치권). 속성: securedDebt, property, priority, maxAmount.  
* LegalEvent: 법률 사건. 하위 유형: LawsuitFiled(소제기), ContractSigned(계약체결), PaymentMade(변제), ForeclosureAuction(경매), ContractRescission(계약해제). 속성: date, participants, outcome.  
* Document: 문서. 속성: docType(예: '계약서', '내용증명우편', '등기사항전부증명서'), date, author, recipient, sourcePage(원본 문서 페이지).  
* LegalClaim: 법적 주장. 속성: claimType(예: '채무이행청구', '손해배상청구', '등기말소청구'), claimant, defendant, basis.

#### **관계 유형 (엣지)**

* HAS\_PARTY(당사자이다), HAS\_CREDITOR(채권자이다), HAS\_DEBTOR(채무자이다), HAS\_LAWYER(변호사이다)  
* IS\_SUBJECT\_OF(의 대상이다), IS\_SECURED\_BY(에 의해 담보된다), IS\_LOCATED\_AT(에 위치한다)  
* TRANSFERRED\_TO(에게 양도되다), LEASED\_TO(에게 임대되다), SOLD\_TO(에게 매도되다)  
* ALLEGES(주장하다), DEFENDS\_AGAINST(에 대해 항변하다), EVIDENCED\_BY(에 의해 증명되다)  
* OCCURRED\_ON(에 발생하다), SENT\_ON(에 발송되다), RECEIVED\_ON(에 도달하다)

이러한 온톨로지 스키마는 단순한 데이터 분류 체계를 넘어, 법률적 사실관계의 논리적 일관성을 강제하는 역할을 수행합니다. 예를 들어, Mortgage 노드는 반드시 Debt 노드와 Property 노드에 연결되어야 한다는 규칙을 정의함으로써, 피담보채권 없는 근저당권이라는 법리적으로 성립 불가능한 정보가 추출되는 것을 원천적으로 방지합니다. 이처럼 온톨로지 자체에 법률적 제약 조건을 내재시키는 것은 LLM의 환각을 억제하고, 추출된 지식의 법적 타당성을 보장하는 첫 번째 방어선이 됩니다.

### **1.2 LLM 기반 정보 추출 및 삼중항 변환**

정의된 스키마를 바탕으로, LLM은 소송기록 텍스트를 분석하여 개체와 관계의 인스턴스를 식별하고 이를 구조화된 삼중항(triple) 형태, 즉 (주어, 서술어, 목적어)로 변환합니다. 이 과정은 기록에 산재된 5개의 분쟁 각각에 대해 수행됩니다.

#### **1.2.1 분쟁 1: '해드림' 인테리어 자재대금 채권**

* **관련 문서:** 의뢰인 상담일지, 최고 통지서, 영업양도 계약서, 이행각서, 오민한의 통지서.  
* **추출된 삼중항 (예시):**  
  * (강용원, HAS\_CREDITOR\_OF, 김선웅)  
  * (Debt\_HaeDream, principalAmount, "2억 원")  
  * (Debt\_HaeDream, is\_subject\_of, SalesAgreement\_20210605)  
  * (SalesAgreement\_20210605, occurredOn, "2021-06-05")  
  * 1  
  * (ContentMail\_20240504, isEvent, InterruptionOfPrescription\_Attempt)  
  * (PromissoryNote\_20240805, isEvent, AcknowledgementOfDebt)  
  * (PromissoryNote\_20240805, performedBy, 김선웅)  
  * (BusinessTransfer\_20240705, transferor, 김선웅)  
  * (BusinessTransfer\_20240705, transferee, 오민한)  
  * (BusinessTransfer\_20240705, usesTradeName, "해드림(SUN-DREAM)")  
  * (BusinessTransfer\_20240705, hasClause, "을은 본 계약일 전에 갑의 영업으로 인해 발생한 채무에 대하여 책임을 지지 아니한다.")

이 분쟁은 복수의 법적 쟁점을 내포합니다. 영업양수인 오민한은 '해드림'이라는 상호를 계속 사용하고 있어 상법 제42조에 따른 영업양수인의 책임이 문제됩니다.2 그러나 양수도 계약서에는 명시적인 면책 조항이 존재합니다.1 또한, 2021년 6월 5일에 발생한 물품대금 채권은 3년의 단기소멸시효가 적용될 수 있습니다.5 이에 대해 채권자 강용원은 내용증명(최고)을 발송하였고 1, 결정적으로 영업양도인 김선웅이 영업양도

*이후*인 2024년 8월 5일에 이행각서를 작성하여 채무를 승인했습니다.1 채무승인은 소멸시효를 중단시키는 효력이 있습니다.7 여기서 핵심 쟁점은 양도인의 채무승인이 양수인에게도 효력을 미치는지 여부입니다. 판례는 상호속용 양수인의 책임과 양도인의 책임은 부진정연대채무 관계에 있어, 일방에 대한 시효중단의 효력이 타방에게 미치지 않는다고 봅니다.9 지식 그래프는 각 사건(영업양도, 채무승인)을 날짜가 명기된 별개의 노드로 생성함으로써, 이러한 법률 관계를 시간 순서에 따라 정확하게 분석할 수 있는 구조를 제공합니다.

#### **1.2.2 분쟁 2: 성수동 대지 및 무효인 경매**

* **관련 문서:** 의뢰인 상담일지, 등기사항전부증명서, 차용증 2건, 대한은행 답신, 이문호-박성희 계약서, 감정평가 의견서.
* **추출된 삼중항 (예시):**  
  * (Loan\_1, principalAmount, "2억 원"), (Loan\_1, interestRate, "10%"), (Loan\_1, dueDate, "2024-01-05")  
  * (Loan\_2, principalAmount, "3억 원"), (Loan\_2, interestRate, "15%"), (Loan\_2, dueDate, "2024-07-05")  
  * (Mortgage\_1, secures, Loan\_1), (Mortgage\_1, onProperty, SeongsuLand\_2/5\_share)  
  * (Mortgage\_2, secures, Loan\_2), (Mortgage\_2, onProperty, SeongsuLand\_3/5\_share)  
  * (Payment\_20240705, amount, "4억 원"), (Payment\_20240705, from, 강용원), (Payment\_20240705, to, 오민한), (Payment\_20240705, hasAttribute, "NoSpecificDesignation")  
  * (ForeclosureAuction\_20240721, initiatedBy, 오민한), (ForeclosureAuction\_20240721, basedOn, Mortgage\_1), (ForeclosureAuction\_20240721, basedOn, Mortgage\_2)  
  * (이문호, acquired, SeongsuLand\_Property), (이문호, through, ForeclosureAuction\_20240721)  
  * (대한은행, HAS\_SECURITY\_INTEREST, Mortgage\_3), (Mortgage\_3, onProperty, SeongsuLand\_Property), (Mortgage\_3, debtor, 이문호)

성수동 대지 분쟁의 모든 법률관계는 2024년 7월 5일 강용원이 오민한에게 지급한 4억 원의 변제 효력에 달려 있습니다.1 지식 그래프는 이 사건의 논리적 연쇄 효과를 명확히 보여줍니다. 변제 당시 충당할 채무를 특정하지 않았으므로("NoSpecificDesignation" 속성), 민법 제477조의 법정변제충당 규정이 적용됩니다.11 이 규정에 따르면 채무자에게 변제 이익이 많은 채무, 즉 이자율이 더 높은 채무부터 변제에 충당되어야 합니다. 따라서 4억 원은 이자율 15%인 3억 원 채무(Loan\_2)의 원리금 전액을 소멸시키고, 남은 금액이 10% 이자율의 2억 원 채무(Loan\_1)에 충당됩니다. 이로 인해 3억 원 채무를 담보하던 근저당권(Mortgage\_2)은 피담보채무 소멸로 무효가 됩니다. 그럼에도 오민한은 이 무효인 근저당권에 기해 경매를 신청하였으므로 1, 해당 경매 절차는 원인 무효입니다.13 우리 법제는 부동산 등기의 공신력을 인정하지 않으므로 15, 무효인 경매를 통해 소유권을 취득한 이문호의 소유권이전등기는 원인무효가 됩니다. 나아가 이문호의 무효인 소유권을 기초로 설정된 대한은행의 근저당권 역시 무효가 됩니다. 지식 그래프는 이처럼 하나의 법률 행위(변제)가 후속 법률 관계(근저당권의 효력, 경매의 유효성, 소유권 이전, 새로운 근저당권 설정)에 미치는 도미노 효과를 시각적, 논리적으로 명확하게 표현합니다.

#### **1.2.3 분쟁 3: 신림동 상가와 사해행위**

* **관련 문서:** 의뢰인 상담일지, 내용증명, 차용증, 등기부등본, 감정평가서, 오민한 답신, 각서.
* **추출된 삼중항 (예시):**  
  * (강용원, HAS\_CREDITOR\_OF, 오국한)  
  * (Debt\_Ogh\_Total, principalAmount, "2억 원")  
  * (SalesAgreement\_Sinlim, seller, 오국한), (SalesAgreement\_Sinlim, buyer, 오민한), (SalesAgreement\_Sinlim, property, SinlimProperty), (SalesAgreement\_Sinlim, occurredOn, "2023-03-17")  
  * (SinlimProperty, hasAttribute, "Debtor's\_Sole\_Significant\_Asset")  
  * (오민한, ALLEGES\_DEFENSE, "ContractTitleTrust")  
  * (ContractTitleTrust\_Claim, evidencedBy, 각서\_20150917)

오민한은 오국한으로부터 신림동 상가를 이전받은 행위가 2015년의 계약명의신탁(Contract Title Trust) 관계를 해소한 것에 불과하므로 사해행위가 아니라고 주장합니다.1 그의 주장은 2015년 작성된 각서를 통해 뒷받침됩니다.1 그러나 시스템의 법률 지식 베이스는 이러한 주장의 허점을 정확히 포착합니다. 대법원 판례에 따르면, 매도인이 명의신탁 약정 사실을 알지 못하는 계약명의신탁의 경우, 명의수탁자(오국한)는 해당 부동산의 완전한 소유권을 취득하며, 그 부동산은 수탁자의 일반 채권자들을 위한 책임재산이 됩니다.18 명의신탁자(오민한)는 수탁자에 대해 부동산 자체가 아닌 매수자금 상당의 부당이득반환채권만을 가질 뿐입니다. 따라서 2023년 채무초과 상태의 수탁자 오국한이 신탁자 오민한에게 부동산 소유권을 이전한 행위는, 신탁자 본인의 재산을 되찾아가는 행위가 아니라 수탁자의 책임재산을 감소시키는 새로운 처분행위로서 다른 채권자들에 대한 사해행위에 해당합니다.18 지식 그래프는 오민한의 주장(명의신탁 해소)과 그 근거(각서), 그리고 이를 반박하는 지배적인 법리(판례)를 나란히 구조화함으로써, 방어 논리의 법리적 취약점을 명확히 드러냅니다.

#### **1.2.4 분쟁 4: 평택시 빌라와 유치권자의 의무**

* **관련 문서:** 의뢰인 상담일지, 등기부등본, 가족관계증명서 & 상속포기 심판, 소멸청구 통지서, 박광윤 답변서, 공사도급 계약서, 임료시세확인서.
* **추출된 삼중항 (예시):**  
  * (박광윤, HAS\_SECURITY\_INTEREST, Lien\_Pyeongtaek)  
  * (Lien\_Pyeongtaek, securedDebt, Debt\_Construction\_3억)  
  * (Lien\_Pyeongtaek, onProperty, PyeongtaekVilla)  
  * (박광윤, performedAction, UnauthorizedLease), (UnauthorizedLease, period, "2022-02-01 to 2022-07-31")  
  * (박광윤, performedAction, PersonalResidence), (PersonalResidence, period, "2022-08-01 to present")  
  * (강용원, isHeirOf, 강호연), (양정숙, isHeirOf, 강호연)  
  * (LienExtinguishmentDemand\_20240829, sentBy, 강용원), (LienExtinguishmentDemand\_20240829, sentBy, 양정숙)

유치권자 박광윤은 소유자의 동의 없이 유치물을 제3자에게 임대하고, 개인적인 주거 공간으로 사용함으로써 선량한 관리자의 주의의무(선관주의의무)를 위반했습니다.1 이는 소유자에게 유치권 소멸청구권을 부여합니다. 여기서 쟁점은 건물의 새로운 소유자(상속인 강용원, 양정숙)가 소멸청구를 할 수 있는지 여부인데, 판례는 양수인도 소멸청구를 할 수 있음을 명확히 하고 있습니다.23 박광윤은 임대 수익을 피담보채권에 충당했다고 주장하지만 1, 이는 법리적으로 타당하지 않습니다. 유치권자가 유치물의 과실(果實)을 수취할 권리(민법 제323조)는 적법한 사용을 전제로 하는데 22, 무단 임대는 위법한 행위이므로 그로 인한 수익은 정당한 과실이 아닌 소유자에게 반환해야 할 부당이득에 해당합니다.25 지식 그래프는 박광윤의 행위를 'UnauthorizedLease'와 'PersonalResidence'라는 별개의 노드로 구분하고, 이를 선관주의의무 위반 및 부당이득 발생과 연결함으로써, 그의 주장이 왜 법적으로 성립할 수 없는지를 논리적으로 설명합니다.

#### **1.2.5 분쟁 5: 흑석동 상가 임대차와 계약 해제의 효력**

* **관련 문서:** 의뢰인 상담일지, 부동산 매매계약서, 임대차계약서, 사업자등록증, 통지서 2건, 해제 통지서. 
* **추출된 삼중항 (예시):**  
  * (SalesAgreement\_Heukseok, seller, 윤건우), (SalesAgreement\_Heukseok, buyer, 이수인)  
  * (이수인, receivedPossessionOf, HeukseokProperty), (이수인, onDate, "2023-10-01")  
  * (LeaseAgreement\_Heukseok, landlord, 이수인), (LeaseAgreement\_Heukseok, tenant, 양정숙)  
  * (양정숙, perfectedLeaseholdRights, HeukseokProperty), (perfectedLeaseholdRights, through, "Building\_Possession" and "Business\_Registration"), (perfectedLeaseholdRights, onDate, "2024-02-01")  
  * (SalesAgreement\_Heukseok, wasSubjectOf, RescissionEvent\_20240515)  
  * (RescissionEvent\_20240515, initiatedBy, 윤건우), (RescissionEvent\_20240515, dueTo, "Buyer's\_Default")

이 분쟁은 임차인의 대항력이 매도인과 매수인 사이의 계약 해제 효과에 우선할 수 있는지를 묻습니다. 임차인 양정숙은 매수인 이수인으로부터 상가를 인도받고 사업자등록을 마침으로써 상가건물 임대차보호법상의 대항력을 취득했습니다.1 핵심 법리는 계약 해제의 효과로부터 제3자를 보호하는 민법 제548조 제1항 단서입니다. 대법원 판례는 매매계약이 해제되기 전에, 매수인으로부터 목적물을 임차하여 주택의 인도와 주민등록(상가건물의 경우 사업자등록)을 마침으로써 대항요건을 갖춘 임차인은 계약 해제로 권리를 침해받지 않는 제3자에 해당한다고 일관되게 판시하고 있습니다.29 지식 그래프는 임차권의 대항력 취득 시점(2024년 2월 1일)과 매매계약의 해제 시점(2024년 5월 15일)을 정확한 타임스탬프가 있는 노드로 생성합니다. 이 시간적 선후 관계를 명확히 함으로써, 시스템은 양정숙의 권리가 계약 해제 이전에 유효하게 성립되었음을 논증하고, 따라서 그녀가 원소유자 윤건우에게 자신의 임차권을 주장할 수 있다는 결론을 도출할 수 있습니다.

### **1.3 구축된 지식 그래프: 통일된 사실의 기반**

위 과정을 통해 5개 분쟁에 걸친 모든 정보는 하나의 통일된 지식 그래프로 통합됩니다. 이 그래프는 더 이상 LLM이 임의로 해석하거나 변경할 수 없는, 시스템의 '단일 진실 공급원(single source of truth)'이자 '장기 기억'으로 기능합니다. 모든 노드와 관계는 원본 문서의 특정 페이지와 구절에 명시적으로 연결되어 있어, 완전한 감사 가능성을 확보합니다.

#### **그래프 시각화**

구축된 지식 그래프를 시각화하면 사건의 복잡한 관계망을 직관적으로 파악할 수 있습니다. 예를 들어, 강용원 노드는 중심적인 원고로서 ALLEGES(주장하다) 엣지를 통해 오민한, 이문호, 주식회사 대한은행 등 다수의 피고 노드와 연결됩니다. 특히 오민한 노드는 '해드림' 영업양수, '성수동 대지' 경매, '신림동 상가' 사해행위 등 여러 분쟁의 중심에 위치한 허브 노드로 시각화되어, 그가 사건 전체에서 차지하는 중추적 역할을 한눈에 보여줍니다.

#### **마스터 개체 및 자산 대장**

그래프 시각화가 관계를 보여주는 데 탁월하다면, 표 형식의 대장은 특정 개체의 속성을 신속하게 참조하는 데 유용합니다. 다음 표는 지식 그래프의 핵심 데이터를 요약하여 사용자가 사건의 주요 행위자와 자산을 한눈에 파악할 수 있도록 돕는 '색인' 역할을 합니다.

| 개체/자산 ID (내부 노드 ID) | 명칭 | 유형 | 주요 일자 | 주요 금전 가치 | 관련 문서 (출처) |
| :---- | :---- | :---- | :---- | :---- | :---- |
| Person\_KangYongwon | 강용원 | Person | 1947-12-11 | \- | 의뢰인 상담일지 1 |
| Person\_OhMinhan | 오민한 | Person | 1954-06-07 | \- | 영업양도 계약서 1 |
| Person\_LeeMunho | 이문호 | Person | 1964-10-03 | \- | 등기사항전부증명서 1 |
| Corporation\_DaehanBank | 주식회사 대한은행 | Corporation | \- | \- | 등기사항전부증명서 1 |
| RealEstate\_SeongsuLand | 성수동 대지 | RealEstate\_Land | \- | \- | 별지 목록 1 |
| RealEstate\_SinlimStore | 신림동 상가 | RealEstate\_Building | \- | 2억 4,000만 원 (2023년 시가) | 별지 목록 1 |
| RealEstate\_PyeongtaekVilla | 평택시 빌라 | RealEstate\_Condo | \- | \- | 별지 목록 1 |
| RealEstate\_HeukseokStore | 흑석동 상가 | RealEstate\_Condo | \- | 10억 원 (매매대금) | 별지 목록 1 |
| Debt\_HaeDream | 해드림 물품대금 채무 | Debt | 2021-06-05 | 2억 원 | 의뢰인 상담일지 1 |
| Loan\_Seongsu\_1 | 성수동 1차 차용금 | LoanAgreement | 2023-01-06 | 2억 원 | 차용증 1 |
| Loan\_Seongsu\_2 | 성수동 2차 차용금 | LoanAgreement | 2023-07-06 | 3억 원 | 차용증 1 |
| Payment\_20240705 | 4억 원 변제 | PaymentMade | 2024-07-05 | 4억 원 | 의뢰인 상담일지 1 |
| Foreclosure\_Seongsu | 성수동 대지 경매 | ForeclosureAuction | 2024-07-21 | \- | 등기사항전부증명서 1 |
| SalesAgreement\_Sinlim | 신림동 상가 매매 | SalesAgreement | 2023-03-17 | 1억 5,000만 원 (거래가액) | 등기사항전부증명서 1 |
| Lien\_Pyeongtaek | 평택시 빌라 유치권 | Lien | \- | 3억 원 (피담보채권) | 공사도급 계약서 1 |
| SalesAgreement\_Heukseok | 흑석동 상가 매매 | SalesAgreement | 2023-10-01 | 10억 원 | 부동산 매매계약서 1 |

---

## **제2부: 2단계 \- 그래프 기반의 설명 가능한 추론 (상징 → 신경망)**

이 단계는 아키텍처의 후반부로서, 구축된 상징적 지식 그래프를 활용하여 LLM의 추론 과정을 안내하고 근거를 제시함으로써, 신뢰할 수 있고 투명한 법률 분석 결과를 생성하는 과정을 보여줍니다. 이른바 Graph-RAG(검색 증강 생성) 기법이 핵심적인 역할을 합니다.

### **2.1 질의 공식화 및 분해**

사용자의 자연어 질의를 시스템이 처리 가능한 논리적 하위 질문으로 분해하는 것부터 시작합니다. 본 연구에서는 의뢰인의 희망사항 중 성수동 대지에 관한 부분을 질의로 사용합니다: "강용원은 성수동 대지와 관련하여 자신이 가지는 모든 권리를 실현하기 위한 소의 제기를 희망한다.".1

시스템은 이 포괄적인 질의를 다음과 같은 구체적인 하위 질문으로 분해합니다.

1. 현재 성수동 대지에 이해관계를 가진 당사자는 누구인가?  
2. 해당 토지에 대한 소유권 및 담보권의 이전 내역은 어떻게 되는가?  
3. 소유권을 이전시킨 경매 절차는 유효했는가?  
4. 강용원은 각 당사자에 대해 어떠한 법적 청구를 할 수 있는가?

### **2.2 지식 그래프로부터의 하이브리드 검색**

분해된 질문에 답하기 위해, 시스템은 지식 그래프에서 관련 정보를 검색하여 LLM에게 제공할 '컨텍스트'를 구축합니다. 이 검색은 구조적 탐색과 의미론적 확장의 두 단계로 이루어집니다.

#### **1단계: 구조적 그래프 탐색 (누가, 무엇을)**

시스템은 강용원 노드와 성수동 대지 노드를 시작점으로 하여, 이들과 직접적 또는 간접적으로 연결된 모든 노드와 관계를 찾는 형식적 그래프 쿼리(예: Cypher)를 실행합니다.

Cypher

MATCH (p:Person {name: '강용원'})-\[r\*1..4\]-(n)  
WHERE id(n) \= id\_of\_SeongsuLand OR 'Seongsu' IN n.name  
RETURN p, r, n

이 쿼리는 강용원, 오민한에 대한 두 건의 대여금 계약, 두 개의 근저당권, 4억 원 변제 사건, 경매 사건, 새로운 소유자 이문호, 이문호의 대한은행 대출 및 대한은행의 근저당권 등 성수동 대지 분쟁과 관련된 모든 사실관계를 포함하는 하위 그래프(subgraph)를 정확하게 추출합니다.

#### **2단계: 의미론적 및 규칙 기반 확장 (왜, 어떻게)**

다음으로, 시스템은 추출된 하위 그래프의 노드들을 분석하여 관련 법률 원칙을 추가로 검색합니다. '지정 없는 변제(PaymentMade with NoSpecificDesignation)', '경매(ForeclosureAuction)' 등의 노드를 식별하고, 시스템 내부의 법률 지식 베이스에서 이와 관련된 규칙 노드를 찾아 컨텍스트에 추가합니다.

* LegalRule:민법 제477조 (법정변제충당) 11  
* LegalPrinciple:경매무효 (Invalidity of Foreclosure) (피담보채무 소멸 시) 13  
* LegalPrinciple:등기무공신력 (Non-credence of Registration) (부동산 등기) 15

이 과정을 통해 최종적으로 LLM에게 제공될 컨텍스트는 사건의 사실관계뿐만 아니라, 해당 사실관계를 해석하는 데 필요한 법률 규칙까지 모두 포함하는 완전하고 정밀한 정보의 집합이 됩니다.

### **2.3 컨텍스트 증강 및 엄격한 프롬프트 엔지니어링**

검색된 하위 그래프는 JSON과 같은 구조화된 형식으로 직렬화되어 LLM에게 전달될 프롬프트에 삽입됩니다. 이때 사용되는 프롬프트는 LLM의 자유로운 상상력을 통제하고, 오직 제공된 데이터에만 기반하여 추론하도록 강제하는 엄격한 지침을 포함합니다.

### **지시사항**

당신은 법률 분석 종합기입니다. 당신의 임무는 아래 '컨텍스트' 섹션에 제공된 정보만을 사용하여 사용자의 질의에 답변하는 것입니다. 외부 지식을 절대 사용하지 마십시오. 컨텍스트에 존재하지 않는 사실을 추론하지 마십시오. 답변에 포함되는 모든 사실적 진술에 대해, 반드시 형식으로 출처를 명시해야 합니다. 당신의 분석은 논리적 추론의 사슬을 따라야 합니다.

### **컨텍스트**

\[... 2.2절에서 검색된 하위 그래프의 직렬화된 데이터가 여기에 삽입됩니다...\]  
예시:  
{ "node\_id": "Loan\_2", "type": "LoanAgreement", "debtor": "강용원", "creditor": "오민한", "principal": "300,000,000 KRW", "interest\_rate": "15%", "date": "2023-07-06" }  
{ "node\_id": "LegalRule\_Art477", "type": "LegalRule", "citation": "민법 제477조", "text": "당사자가 변제에 충당할 채무를 지정하지 아니한 때에는...채무자에게 변제이익이 많은 채무의 변제에 충당한다..." }  
{ "node\_id": "LegalPrinciple\_NoCredence", "type": "LegalPrinciple", "text": "부동산 등기에는 공신력이 인정되지 않아, 원인무효의 등기에 기초한 후속 등기는 선의의 제3자라도 보호받지 못한다." }

### **사용자 질의**

"제공된 컨텍스트를 바탕으로, 강용원이 성수동 대지에 관한 자신의 모든 권리를 실현하기 위해 제기할 수 있는 모든 법적 청구를 각 청구의 피고와 법적 근거를 명시하여 설명하시오."

### **답변**

이러한 '엄격한 프롬프트'는 LLM의 역할을 근본적으로 변화시킵니다. LLM은 더 이상 지식을 '기억'하거나 '탐색'하는 주체가 아니라, 주어진 사실들을 논리적으로 '종합'하고 '설명'하는 역할에 한정됩니다. 모든 주장에 출처를 명시하도록 강제함으로써, 생성된 답변의 모든 요소는 검증 가능한 지식 그래프의 노드로 역추적될 수 있습니다. 이는 환각의 위험을 극적으로 낮추고, 최종 결과물의 완전한 감사 가능성을 보장합니다.

### **2.4 LLM 생성 분석 및 검증 가능한 인용**

엄격한 프롬프트에 따라, LLM은 다음과 같이 구조화되고 모든 사실관계에 대한 출처가 명시된 분석 결과를 생성합니다.

**제목: 강용원의 성수동 대지 관련 권리 실현을 위한 법적 청구 분석**

**가. 이문호에 대한 청구:**

* **청구 내용:** 소유권이전등기 말소청구  
* **법적 근거:** 이문호가 소유권을 취득한 경매는 무효입니다 \[LegalPrinciple:경매무효\]. 강용원은 2024년 7월 5일 오민한에게 4억 원을 변제하면서 변제할 채무를 특정하지 않았습니다 \[Payment\_20240705\]. 법정변제충당 규정에 따라, 이 변제금은 채무자에게 변제 이익이 가장 큰 채무, 즉 이자율 15%의 3억 원 대출금 \[Loan\_2\]에 우선적으로 충당되어야 합니다. 이 변제로 인해 두 번째 근저당권 \[Mortgage\_2\]의 피담보채무는 소멸하였습니다. 따라서 소멸된 근저당권에 기하여 진행된 경매는 원인무효입니다. 한국 부동산 등기에는 공신력이 인정되지 않으므로 \[LegalPrinciple\_NoCredence\], 무효인 경매를 통해 이루어진 이문호 명의의 소유권이전등기는 효력이 없습니다.

**나. 주식회사 대한은행에 대한 청구:**

* **청구 내용:** 근저당권설정등기 말소청구  
* **법적 근거:** 대한은행의 근저당권 \[Mortgage\_3\]은 이문호의 소유권을 전제로 설정되었습니다. 위에서 설명한 바와 같이 이문호의 소유권 등기가 원인무효이므로, 무효인 소유권에 터 잡아 설정된 대한은행의 근저당권 역시 법률상 원인이 없는 무효의 등기입니다.

**다. 박성희에 대한 청구:**

* **청구 내용:** 성수동 건물 철거 및 부지 인도 청구  
* **법적 근거:** 박성희가 토지와 건물을 점유할 권리는 이문호와의 계약 \[Contract\_Imunho\_Parkseonghee\]에서 비롯됩니다. 이문호는 토지에 대한 적법한 소유권이 없으므로, 박성희에게 유효한 임차권을 설정해 줄 수 없습니다. 또한, 이문호는 토지의 정당한 소유자가 아니므로 그가 신축한 건물은 강용원의 토지 위에 무단으로 건축된 것입니다. 따라서 토지의 진정한 소유자인 강용원은 무단 건축물의 철거와 그 점유자의 퇴거를 청구할 수 있습니다.

**라. 오민한에 대한 청구:**

* **청구 내용:** 불법행위로 인한 손해배상청구  
* **법적 근거:** 오민한은 4억 원의 변제 \[Payment\_20240705\]로 인해 자신의 근저당권 \[Mortgage\_2\]이 소멸되었음을 알았거나 알 수 있었음에도 불구하고, 고의 또는 과실로 무효인 담보권에 기하여 경매를 실행했습니다 \[ForeclosureAuction\_20240721\]. 이 위법한 담보권 실행 행위로 인해 강용원은 소유권을 상실하고 법적 분쟁 비용을 지출하는 등의 손해를 입었으므로, 이는 불법행위를 구성합니다.

---

## **결론: 신뢰 가능한 법률 AI를 위한 신경망 인식과 상징 논리의 통합**

본 보고서는 복잡한 민사소송기록을 대상으로 신경-상징 하이브리드 아키텍처를 적용하는 전 과정을 시연했습니다. 1단계에서 LLM의 강력한 자연어 이해 능력을 활용하여 비정형 기록으로부터 정밀한 법률 지식 그래프를 구축하고, 2단계에서는 이 지식 그래프를 '단일 진실 공급원'으로 삼아 LLM의 추론을 엄격하게 통제함으로써, 사실에 기반하고 설명 가능한 분석 결과를 생성했습니다.

이 아키텍처가 제시하는 핵심적인 이점은 다음과 같습니다.

* **환각 현상 억제:** LLM의 추론은 검증 가능한 지식 그래프의 사실에 엄격히 구속됩니다. LLM은 더 이상 사실을 '기억'하지 않고, 주어진 사실을 '종합'하는 역할만 수행합니다.  
* **내재된 설명가능성:** 최종 결과물의 모든 진술은 지식 그래프의 특정 노드로 역추적이 가능하며, 그 노드는 다시 원본 문서의 특정 구절로 연결됩니다. 이는 AI의 판단 근거를 투명하게 제시하여 '블랙박스' 문제를 해결합니다.  
* **전체론적 분석:** 지식 그래프 구조는 일견 무관해 보이는 분쟁들 사이의 숨겨진 연결고리(예: 오민한의 중추적 역할)를 자연스럽게 드러내어, 단순한 텍스트 순차 독해로는 파악하기 어려운 전체적인 사건 구조에 대한 깊이 있는 이해를 가능하게 합니다.

결론적으로, 신경망의 유연한 '인식' 능력과 상징 시스템의 엄격한 '논리'를 결합하는 본 신경-상징 아키텍처는, 강력한 성능과 더불어 법률 전문가들이 요구하는 최고 수준의 신뢰성, 투명성, 책임성을 담보할 수 있는 차세대 법률 AI 도구를 구축하는 현실적인 청사진을 제시합니다. 이는 법률 실무의 신뢰를 얻을 수 있는 인공지능 시스템 개발의 중요한 이정표가 될 것입니다.

#### **참고 자료**

1. @민사법(기록형)-최종본.pdf  
2. 상법 제42조 제1항에 따른 상호속용 영업양수인의 책임이 문제되는 사건\[대법원 2023\. 12\. 7\. 선고 중요 판결\] \- 판례속보 \- 대한민국 법원, 6월 23, 2025에 액세스, [https://www.scourt.go.kr/portal/news/NewsViewAction.work?pageIndex=1\&searchWord=\&searchOption=\&seqnum=9611\&gubun=4\&type=5](https://www.scourt.go.kr/portal/news/NewsViewAction.work?pageIndex=1&searchWord&searchOption&seqnum=9611&gubun=4&type=5)  
3. 영업양수인의 상호속용에 관한 판례 정리 \- 윤정근변호사 \- 뉴스 & 자료 :: 법무법인, 6월 23, 2025에 액세스, [https://daraelaw.co.kr/board/news/1506](https://daraelaw.co.kr/board/news/1506)  
4. 상호를 속용하는 영업양수인의 연대책임 \- 법무법인 반율, 6월 23, 2025에 액세스, [http://thelaw.co.kr/%EC%83%81%ED%98%B8%EB%A5%BC-%EC%86%8D%EC%9A%A9%ED%95%98%EB%8A%94-%EC%98%81%EC%97%85%EC%96%91%EC%88%98%EC%9D%B8%EC%9D%98-%EC%97%B0%EB%8C%80%EC%B1%85%EC%9E%84/](http://thelaw.co.kr/%EC%83%81%ED%98%B8%EB%A5%BC-%EC%86%8D%EC%9A%A9%ED%95%98%EB%8A%94-%EC%98%81%EC%97%85%EC%96%91%EC%88%98%EC%9D%B8%EC%9D%98-%EC%97%B0%EB%8C%80%EC%B1%85%EC%9E%84/)  
5. 물품대금청구소송의 요건과 절차, 소멸시효는? | 대륜 \- 손해배상, 6월 23, 2025에 액세스, [https://www.daeryunlaw-compensation.com/lawInfo\_new/6703](https://www.daeryunlaw-compensation.com/lawInfo_new/6703)  
6. 코아채권관리 연구소, 6월 23, 2025에 액세스, [https://www.corera.co.kr/community/community04.php?show=view\&id=106\&offset=0\&board=community04](https://www.corera.co.kr/community/community04.php?show=view&id=106&offset=0&board=community04)  
7. 법무법인 공유 채권회수센터 연구소, 6월 23, 2025에 액세스, [http://www.chusim.co.kr/v2/about/view.asp?seq=90\&page=7\&SearchWord=\&SearchString=](http://www.chusim.co.kr/v2/about/view.asp?seq=90&page=7&SearchWord&SearchString)  
8. 소멸시효 중단사유 \- 승인(제168조 제3호) | 민법총칙 \- 네플라, 6월 23, 2025에 액세스, [https://www.nepla.ai/wiki/%EB%AF%BC%EC%82%AC/%EB%AF%BC%EB%B2%95%EC%B4%9D%EC%B9%99/%EC%86%8C%EB%A9%B8%EC%8B%9C%ED%9A%A8-%EC%A4%91%EB%8B%A8%EC%82%AC%EC%9C%A0-%EC%8A%B9%EC%9D%B8-%EC%A0%9C168%EC%A1%B0-%EC%A0%9C3%ED%98%B8-6349gygm9zx5](https://www.nepla.ai/wiki/%EB%AF%BC%EC%82%AC/%EB%AF%BC%EB%B2%95%EC%B4%9D%EC%B9%99/%EC%86%8C%EB%A9%B8%EC%8B%9C%ED%9A%A8-%EC%A4%91%EB%8B%A8%EC%82%AC%EC%9C%A0-%EC%8A%B9%EC%9D%B8-%EC%A0%9C168%EC%A1%B0-%EC%A0%9C3%ED%98%B8-6349gygm9zx5)  
9. 상호를 속용하는 영업양수인의 책임 \- 법무법인 우리누리, 6월 23, 2025에 액세스, [https://woorinurilaw.com/2025/01/22/%EC%83%81%ED%98%B8%EB%A5%BC-%EC%86%8D%EC%9A%A9%ED%95%98%EB%8A%94-%EC%98%81%EC%97%85%EC%96%91%EC%88%98%EC%9D%B8%EC%9D%98-%EC%B1%85%EC%9E%84/](https://woorinurilaw.com/2025/01/22/%EC%83%81%ED%98%B8%EB%A5%BC-%EC%86%8D%EC%9A%A9%ED%95%98%EB%8A%94-%EC%98%81%EC%97%85%EC%96%91%EC%88%98%EC%9D%B8%EC%9D%98-%EC%B1%85%EC%9E%84/)  
10. 소멸시효/중단 \- 나무위키, 6월 23, 2025에 액세스, [https://namu.wiki/w/%EC%86%8C%EB%A9%B8%EC%8B%9C%ED%9A%A8/%EC%A4%91%EB%8B%A8](https://namu.wiki/w/%EC%86%8C%EB%A9%B8%EC%8B%9C%ED%9A%A8/%EC%A4%91%EB%8B%A8)  
11. 조문정보 | 국가법령정보센터, 6월 23, 2025에 액세스, [http://www.law.go.kr/LSW//lsLawLinkInfo.do?lsJoLnkSeq=900738810\&chrClsCd=010202](http://www.law.go.kr/LSW//lsLawLinkInfo.do?lsJoLnkSeq=900738810&chrClsCd=010202)  
12. 법정충당 (민법 제477조), 6월 23, 2025에 액세스, [https://brunch.co.kr/@startlrah/354](https://brunch.co.kr/@startlrah/354)  
13. 부당이득금\[무효인 부동산 임의경매에 따라 수령한 배당금에 관해 부당이득반환을 청구한 사건\], 6월 23, 2025에 액세스, [https://www.law.go.kr/LSW/precInfoP.do?mode=0\&precSeq=236819](https://www.law.go.kr/LSW/precInfoP.do?mode=0&precSeq=236819)  
14. \[칼럼\] 허위의 근저당권 설정과 사기죄 \- 부동산태인, 6월 23, 2025에 액세스, [https://www.taein.co.kr/community/consult/column/column\_board\_viewV2.php?num=654](https://www.taein.co.kr/community/consult/column/column_board_viewV2.php?num=654)  
15. \[부동산돋보기\] '등기' 무늬만 공신력 ... 시장 불안정 심화의 주범, 6월 23, 2025에 액세스, [https://www.karnews.or.kr/news/articleView.html?idxno=18876](https://www.karnews.or.kr/news/articleView.html?idxno=18876)  
16. 등기의 공신력 \- 나무위키, 6월 23, 2025에 액세스, [https://namu.wiki/w/%EB%93%B1%EA%B8%B0%EC%9D%98%20%EA%B3%B5%EC%8B%A0%EB%A0%A5](https://namu.wiki/w/%EB%93%B1%EA%B8%B0%EC%9D%98%20%EA%B3%B5%EC%8B%A0%EB%A0%A5)  
17. 부동산 등기공신력과 경매의 공신적 효력에 대한 실무상 이해 \- Naver Blog, 6월 23, 2025에 액세스, [https://blog.naver.com/cheong8780/221658705926?viewType=pc](https://blog.naver.com/cheong8780/221658705926?viewType=pc)  
18. 사해행위취소 | 국가법령정보센터 | 판례, 6월 23, 2025에 액세스, [https://www.law.go.kr/LSW/precInfoP.do?precSeq=64767](https://www.law.go.kr/LSW/precInfoP.do?precSeq=64767)  
19. 명의신탁된 부동산처분과 사해행위 | 한국경제, 6월 23, 2025에 액세스, [https://www.hankyung.com/article/202206070307Q](https://www.hankyung.com/article/202206070307Q)  
20. 유치권자의 선관주의의무 위반과 유치권의 소멸청구 : 네이버 블로그, 6월 23, 2025에 액세스, [https://blog.naver.com/yeoamlaw/223070628267?viewType=pc](https://blog.naver.com/yeoamlaw/223070628267?viewType=pc)  
21. 여러 개의 유치물 중 일부에 대하여 선량한 관리자의 의무위반이 인정되는 경우 유치권 소멸청구에 따른 유치권 소멸의 범위가 문제된 사안\[대법원 2022\. 6\. 16\. 선고 중요판결\] \- 판례속보 \- 대한민국 법원, 6월 23, 2025에 액세스, [https://www.scourt.go.kr/portal/news/NewsViewAction.work?seqnum=8547\&gubun=4\&searchOption=\&searchWord=%B0%FC%B8%AE](https://www.scourt.go.kr/portal/news/NewsViewAction.work?seqnum=8547&gubun=4&searchOption&searchWord=%EF%BF%BD%EF%BF%BD%EF%BF%BD%EF%BF%BD)  
22. 유치권자의 유치물 사용과 보존필요 여부 \- 한국경제, 6월 23, 2025에 액세스, [https://www.hankyung.com/article/202206073265Q](https://www.hankyung.com/article/202206073265Q)  
23. 유치권자의 주의의무 \- 건설산업신문, 6월 23, 2025에 액세스, [http://m.cennews.co.kr/news/articleView.html?idxno=20369](http://m.cennews.co.kr/news/articleView.html?idxno=20369)  
24. 유치권 \- 판례 : 네이버 블로그, 6월 23, 2025에 액세스, [https://m.blog.naver.com/syntheticwood/221667839607](https://m.blog.naver.com/syntheticwood/221667839607)  
25. 유치권에 기한 임차인의 임대목적물 사용과 부당이득반환의무 \- 건국대학교 법학연구소, 6월 23, 2025에 액세스, [http://ils.konkuk.ac.kr/board\_common/file\_download.asp?Board\_Key=21\&File\_Key=214\&flag=4](http://ils.konkuk.ac.kr/board_common/file_download.asp?Board_Key=21&File_Key=214&flag=4)  
26. 유치권소멸청구 및 유치권자의 부당이득반환 의무 \- 네이버 블로그, 6월 23, 2025에 액세스, [https://m.blog.naver.com/kmn0930/220979710456](https://m.blog.naver.com/kmn0930/220979710456)  
27. www.easylaw.go.kr, 6월 23, 2025에 액세스, [https://www.easylaw.go.kr/CSP/CnpClsMainBtr.laf?popMenu=ov\&csmSeq=627\&ccfNo=2\&cciNo=3\&cnpClsNo=1\&menuType=onhunqna\#:\~:text=%E2%98%9E%20%EC%9E%84%EB%8C%80%EC%B0%A8%EB%8A%94%20%EA%B7%B8%20%EB%93%B1%EA%B8%B0,%EB%8B%A4%EC%9D%8C%EB%82%A0%EB%B6%80%ED%84%B0%20%EB%8C%80%ED%95%AD%EB%A0%A5%EC%9D%B4%20%EC%83%9D%EA%B9%81%EB%8B%88%EB%8B%A4.\&text=%E2%98%9E%20%EB%8C%80%ED%95%AD%EB%A0%A5%EC%9D%80%20%EC%9E%84%EC%B0%A8%EC%9D%B8%EC%9D%B4,%EC%9E%90%EC%97%90%EA%B2%8C%20%ED%9A%A8%EB%A0%A5%EC%9D%B4%20%EC%83%9D%EA%B9%81%EB%8B%88%EB%8B%A4.](https://www.easylaw.go.kr/CSP/CnpClsMainBtr.laf?popMenu=ov&csmSeq=627&ccfNo=2&cciNo=3&cnpClsNo=1&menuType=onhunqna#:~:text=%E2%98%9E%20%EC%9E%84%EB%8C%80%EC%B0%A8%EB%8A%94%20%EA%B7%B8%20%EB%93%B1%EA%B8%B0,%EB%8B%A4%EC%9D%8C%EB%82%A0%EB%B6%80%ED%84%B0%20%EB%8C%80%ED%95%AD%EB%A0%A5%EC%9D%B4%20%EC%83%9D%EA%B9%81%EB%8B%88%EB%8B%A4.&text=%E2%98%9E%20%EB%8C%80%ED%95%AD%EB%A0%A5%EC%9D%80%20%EC%9E%84%EC%B0%A8%EC%9D%B8%EC%9D%B4,%EC%9E%90%EC%97%90%EA%B2%8C%20%ED%9A%A8%EB%A0%A5%EC%9D%B4%20%EC%83%9D%EA%B9%81%EB%8B%88%EB%8B%A4.)  
28. 보증금의 보호 \< 상가건물 임대차 \- 찾기쉬운 생활법령정보, 6월 23, 2025에 액세스, [https://www.easylaw.go.kr/CSP/CnpClsMainBtr.laf?popMenu=ov\&csmSeq=627\&ccfNo=2\&cciNo=3\&cnpClsNo=1\&menuType=onhunqna](https://www.easylaw.go.kr/CSP/CnpClsMainBtr.laf?popMenu=ov&csmSeq=627&ccfNo=2&cciNo=3&cnpClsNo=1&menuType=onhunqna)  
29. 미등기매수인이 주택임대 후 매매계약 해제시 임차인의 보호 \- 대법원 2023다201218, 201225 판결 : 네이버 블로그, 6월 23, 2025에 액세스, [https://blog.naver.com/lawjwsang/223180119334](https://blog.naver.com/lawjwsang/223180119334)  
30. 임차인은 소유자의 매매계약 해제에도 보호된다는 사례, 6월 23, 2025에 액세스, [http://bpark.co.kr/bbs/board.php?bo\_table=board1\_2\&wr\_id=90](http://bpark.co.kr/bbs/board.php?bo_table=board1_2&wr_id=90)  
31. 건물명도·소유권이전등기 | 국가법령정보센터 | 판례, 6월 23, 2025에 액세스, [https://www.law.go.kr/LSW/precInfoP.do?precSeq=85759](https://www.law.go.kr/LSW/precInfoP.do?precSeq=85759)  
32. 유용한 정보 \- 코아채권관리 연구소, 6월 23, 2025에 액세스, [https://www.corera.co.kr/renew/customer/customer4.php?show=view\&id=65\&offset=20\&board=community04](https://www.corera.co.kr/renew/customer/customer4.php?show=view&id=65&offset=20&board=community04)  
33. 변제 \- 나무위키, 6월 23, 2025에 액세스, [https://namu.wiki/w/%EB%B3%80%EC%A0%9C](https://namu.wiki/w/%EB%B3%80%EC%A0%9C)  
34. 변제충당 순서에 대하여 | 로톡, 6월 23, 2025에 액세스, [https://www.lawtalk.co.kr/posts/100079](https://www.lawtalk.co.kr/posts/100079)  
35. 지정충당 설명을 : 네이버 블로그, 6월 23, 2025에 액세스, [https://blog.naver.com/startlrah/222709910014?viewType=pc](https://blog.naver.com/startlrah/222709910014?viewType=pc)  
36. 경매무효 시 소유권이전등기말소의무와 배당금반환의무는 동시이행 아니라는 사례, 6월 23, 2025에 액세스, [http://www.bpark.co.kr/bbs/board.php?bo\_table=board5\_3\&wr\_id=18](http://www.bpark.co.kr/bbs/board.php?bo_table=board5_3&wr_id=18)
