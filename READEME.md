# aws-customized-message-bedrock

## 데모 개요
카카오페이손해보험은 현재 고객의 보험 이벤트 발생 시(보험 가입, 보험금 청구, 보험 종료 등) 메시지를 고객에게 카카오톡으로 전송하고 있습니다. 그러나 전달하는 메시지가 표준화된 메시지로, 고객의 컨텍스트를 반영하지 못하고 있습니다. 다양한 보험 이벤트 발생 시 고객의 컨텍스트 정보를 기반으로 LLM을 통해 개인화된 메시지를 생성하고 전달합니다. 예를 들어, 해외여행보험 가입 시 고객이 방문하는 여행지의 예상되는 날씨 정보, 유명한 여행지, 보험 가입 횟수를 반영하여 메시지를 생성합니다.

## 데모 아키텍처

<img src="/architecture.png"></img>

1. 보험 이벤트가 발생하면 사용자 정보 및 이벤트 항목을 전달합니다.
2. 사용자 정보를 기반으로 보험 계약 정보를 가져옵니다. 데모에서는 별도로 RDS를 호출하지 않고 하드코딩으로 구현했습니다.
3. 보험 계약 정보를 기반으로 외부 데이터를 추가로 가져옵니다. 예를 들어 사용자가 해외여행보험을 가입 할 경우 고객이 방문하는 도시와 날짜를 기반으로 예상되는 날씨를 가져옵니다.
4. 사용자 컨텍스트를 기반으로 LLM에 개인화된 메세지를 생성합니다. 데모에서는 프롬프트 엔지니어링 기법으로 Few-shot 기법을 사용했습니다.
    1. Few-shot 프롬프트 기법?
        1. 모델에 원하는 작업의 몇 가지 예시(보통 줄여서 shot이라고 함)를 제시합니다.
5. LLM이 생성한 메세지로 사용자에게 메세지를 전달합니다.

## 데모 결과

**시나리오-1**. 고객은 5월 도쿄를 방문하며 해외여행보험을 세번째로 가입.
<img src="/result1.png"></img>
**시나리오-2** 고객은 베이징을 방문하며 해외여행보험을 처음으로 가입
<img src="/result2.png"></img>
**시나리오-3**. 고객이 여행하는 동안 날씨는 좋았으며 특별한 불편함 없이 다녀옴.
<img src="/result3.png"></img>
**시나리오-4**. 고객이 여행하는 동안 비가 계속 왔으며 여행을 마치고 귀국하는 길에 항공기 지연이 발생함.
<img src="/result4.png"></img>





## 코드
createTalk의 예시이며 사용자가 보험을 가입 할 때 사용한 프롬프트 예시입니다.

```python
def createTalk_SignUp(customerContract, estimateWeather):
    
    bedrock_client = boto3.client('bedrock-runtime', region_name = "us-west-2")

    prompt_template = f"""\n\nHuman: 당신은 사용자의 보험 정보를 기반으로 알람 메세지를 생성하는 AI Assistant 입니다. 
    현재 해야 할 일은 해외여행보험을 가입한 고객에게 가입 감사 메세지를 보내야합니다.메세지의 예시를 알려드리겠습니다. 
    
    주어진 정보는 다음과 같습니다.
    - 이름 : 홍길동
    - 여행지 : 샌프란시스코
    - 예상 날씨 : 흐림
    - 보험 시작일 : 2024-05-01
    - 보험 종료일 : 2024-05-15
    - 보험 가입 횟수 : 1
    
    <example>
    Assistant: 홍길동님! 카카오페이손해보험의 해외여행보험을 가입해주셔서 감사합니다. 방문하시는 샌프란시스코의 예상 날씨는 흐림이니 우산을 꼭 챙기시기 바랍니다.
    샌프란시스코의 주요 여행지는 금문교가 유명합니다. 2주간 즐거운 여행 되시길 바랍니다.
    </example>
    
    주어진 정보와 <example></example>을 활용해서 사용자에게 전달할 메세지를 생성해주세요.
    
    주어진 정보 :
    - 이름 : {customerContract['customerName']}
    - 여행지 : {customerContract['destination']}
    - 예상 날씨 : {estimateWeather}
    - 보험 시작일 : {customerContract['startDate']}
    - 보험 종료일 : {customerContract['endDate']}
    - 보험 가입 횟수 : {customerContract['totalContract']}
    
    \n\nAssistant: 
    """
    
    body = json.dumps({
        "prompt" : prompt_template,
        "temperature" : 0.5,
        "max_tokens_to_sample": 2000,
        "top_p": 0.99,
        "top_k": 250
    })
    
    
    response = bedrock_client.invoke_model(
        body = body,
        modelId='anthropic.claude-v2:1'
        )
        
    result = response['body'].read().decode('utf-8')
    message = json.loads(result)['completion']
    
    return message
    
#메세지 결과
허진성님, 카카오페이손해보험의 해외여행보험을 세번째로 가입해주셔서 진심으로 감사드립니다. 
방문하시는 도쿄의 예상 날씨는 좋음이니 안심하고 여행을 즐기시기 바랍니다. 
도쿄의 주요 관광지로는 시부야, 메이지 신궁, 스카이트리가 있습니다. 5월의 도쿄는 벚꽃도 만개할 것입니다. 
10일간 즐거운 여행 되시길 기원합니다.
```

createTalk의 예시이며 사용자가 여행을 마치고 보험을 종료 할 때 프롬프트 예시입니다.