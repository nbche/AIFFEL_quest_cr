# 플러터로 인공지능 서비스 만들기  - 5분 일기장

# 1. 기획 및 디자인
- **앱 주제 선정 및 기본 기능 정의**
    - 주제: 핸드폰의 주요 기능인 카메라를 이용하여 일상의 기록 (일기) 를 남김
    - 특징: 사진의 내용 (시간, 장소, 묘사) 을 AI 모델이 기록해 줌
      
- **정보 구조도**

    ![Image](https://github.com/user-attachments/assets/517bce9e-1b55-47c7-a1b6-3406dcf1500e)

- **와이어프레임**

    ![Image](https://github.com/user-attachments/assets/8f7cd94b-fb7a-475b-b92a-18fe5c09a224)


# 2. 기본 기능 구현
- **프로젝트 생성 및 기본 구조 설정**
    - Flutter Widget Tree 참조

     ![Image](https://github.com/user-attachments/assets/5618a189-0c2e-404f-bfd0-f855c4eca5f6)
      
- **메인 페이지 UI 구현**
   
    ![Image](https://github.com/user-attachments/assets/b0562c4f-e652-4a2d-8505-2fb09731cd9d)
  
     : Appbar 에는 기능이 부여되어 있지 않음
  
     : Bottom navigation 을 통해 사용자는 페이지간의 이동을 하게됨
  
    
# 3. 네비게이션 및 데이터전달
- **페이지 간 네비게이션 구현**
    - 페이지간의 네비게이션은 Bottom Navigation Bar 를 이용함
      
    ![Image](https://github.com/user-attachments/assets/fc32eee3-4812-4daa-9609-3288a8739de7)
    
  
# 4. 고급 기능 통합
- **디바이스 기능 (예: 카메라, 센서) 통합**
    - 사진 촬영 및 갤러리에서 사진 업로드 기능 (Picture Page)
      
    ![Image](https://github.com/user-attachments/assets/5b1eec30-4f27-4c2e-9db3-213e3dc80302)
  

# 5. API 및 외부 서비스 연동
 - **REST API 또는 외부 서비스 연동**
    - Google Gemini API 를 통해 사진 이미지 내용 묘사함 (페이지 하단 부분)
      
    ![Image](https://github.com/user-attachments/assets/28383cce-73f8-452b-9f26-52abe4f6065e)
       
# 6. Flutter 실구현 (화면 및 비디오)
- 파일 사이즈 제한으로 노션 페이지에 올리겠음

# 회고(참고 링크 및 코드 개선)
  
  - 사진을 묘사하는 Describe_Page 를 Picture_Page 의 하위페이지로 설정함으로써 기능 구현및 코드 생성에 애로사항이 있었음. 페이지의 구조 결정시 코더의 역량을 고려하여야 했음.
    
  - 인공지능 채팅 프로그램을 이용하는 장/단점을 절감한 계기가 됨
     장점: 주어진 과제를 시간의 지체없이 해결하려 함
     단점: 코딩의 일부가 생략된 채로 내용을 전달함 (여러번 수정을 요청함에도 불구하고)
    
      예시1): List<Map<String, dynamic>> (제대로된 코딩), List> (Chat GPT 코딩)
      예시2): State<DescriptionEditor> (제대로된 코딩), State (Chat GPT 코딩)
         
  - Google Gemini API 와의 연결이 (상상했던 것보다) 어렵지 않음을 경험함.
    
  - 사진과 기술된 내용이 다음 페이지 (Describe) 로 옮겨지는 과정 중에 에러가 나서 다음 기능을 구현할 수 없음을 아쉬움이 남지만, (반나절을 노력해도) 에러를 스스로 잡아내지 못한지라, 이 부분은 추후 (역량이 보강된 후) 재검토 해보도록 하겠음
   
