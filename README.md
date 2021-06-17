# docker-gpu-speech-command-api-service
[![Quality gate](https://sonarqube.406.csie.nuu.edu.tw/api/project_badges/quality_gate?project=docker-gpu-hakka-speech-command-api-service-14)](https://sonarqube.406.csie.nuu.edu.tw/dashboard?id=docker-gpu-hakka-speech-command-api-service-14)
tensorflow speech command example simple api service 

## 前置工作流程
* 輸出模型檔案

## 新增模型服務API流程
```plantuml
@startuml
申請網域名稱 -> 建立存放模型資料夾 : 例如申請alpha1.hakka.csie.nuu.edu.tw，則在模型資料夾內建立alpha1名稱的資料夾
建立存放模型資料夾 -> 放入輸出的模型檔案 : .pb為模型檔案，txt為模型分類名稱
放入輸出的模型檔案 -> 為模型建立前端服務API: 修改docker-compose.yaml檔案
@enduml
```
## 專案資料夾目錄結構說明1
```plantuml
@startwbs
+ 專案根目錄
 + app(客語指令辨識API服務)
  + static(healthcheck網頁)
  + template(辨識上傳UI與辨識結果)
  +_ app.py(主要辨識服務)
 + 模型(多個辨識模型)
  + alpha(alpha.hakka.csie.nuu.edu.tw，辨識客家語音alpha模型，前一屆)
   +_ conv_labels_1.txt
   -_ my_frozen_graph1.pb
  + alpha1(alpha.hakka.csie.nuu.edu.tw，辨識客家語音alpha1模型，邱聖君)
  + beta(beta.hakka.csie.nuu.edu.tw，辨識客家語音alpha1模型，前一屆)
  + beta1(beta1.hakka.csie.nuu.edu.tw，辨識客家語音alpha1模型，邱聖君)
  + charlie(charlie.hakka.csie.nuu.edu.tw，辨識客家語音alpha1模型，前一屆)
  + charlie1(charlie1.hakka.csie.nuu.edu.tw，辨識客家語音alpha1模型，邱聖君)
@endwbs
```

## 聊天室常用指令
```
# 建立問題
/docker-gpu-hakka-speech-command-api-service issue create 你想要建立的問題名稱
# 執行工作
/docker-gpu-hakka-speech-command-api-service run 要執行的工作名稱
```

# usage 
`sudo docker-compose up -d --build`

# 主機IP
* `IP`: `120.105.128.209`
* `使用者`: `skynet`
