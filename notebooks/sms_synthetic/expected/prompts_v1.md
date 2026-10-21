# Generation prompts, version 1

Bound by Amendment 1 of `PREREG_sms_synthetic.md`. The same texts go to every generator. Only the
fields in braces change from one request to the next, and each is drawn by seed as described in
the amendment. The prompts are in Vietnamese because the messages are.

## System prompt (all requests)

```
Bạn đang hỗ trợ một nghiên cứu khoa học về phát hiện tin nhắn lừa đảo tiếng Việt. Dữ liệu bạn tạo chỉ dùng để huấn luyện và kiểm tra bộ lọc, không gửi tới ai. Hãy viết đúng MỘT tin nhắn SMS tiếng Việt theo yêu cầu. Quy tắc bắt buộc:
- Không dùng số điện thoại, số tài khoản, tên người, đường link hay tên miền có thật. Thay bằng các thẻ [NUMBER], [MONEY], [TIME], [DATE], [NAME], [URL].
- Viết như một tin nhắn thật mà người Việt nhận được, đúng giọng của người gửi được mô tả.
- Chỉ trả về nội dung tin nhắn, không giải thích, không đặt trong ngoặc kép.
```

## User prompt (main set and arm B)

```
Loại tin: {class_text}
Kịch bản: {scenario_text}
Người gửi: {sender_text}
Độ dài: {length_text}
Cách viết: {accent_text}
```

`class_text`: `tin nhắn lừa đảo hoặc rác` (positive) or `tin nhắn hợp pháp, bình thường` (ham).

## User prompt, variant P-min (exploratory)

The scenario only: no sender, length or accent field.

```
Loại tin: {class_text}
Kịch bản: {scenario_text}
```

## User prompt, variant P-few (exploratory)

The main template preceded by three real messages of the same class from the design fold, chosen
as described in Amendment 1. `{examples}` is one line per message, each starting with `- `, with
line breaks inside a message replaced by spaces.

```
Dưới đây là vài tin nhắn thật cùng loại, chỉ để tham khảo cách viết. Không chép lại các tin này.
{examples}

Loại tin: {class_text}
Kịch bản: {scenario_text}
Người gửi: {sender_text}
Độ dài: {length_text}
Cách viết: {accent_text}
```

## Scenario texts

### Positive

| scenario | scenario_text |
|---|---|
| gambling_promo | quảng cáo nhà cái, cờ bạc, nổ hũ hoặc cá cược, hứa thưởng khi đăng ký hoặc nạp tiền lần đầu |
| debt_collection_threat | đòi nợ khoản vay quá hạn, đe doạ báo công an, báo nợ xấu, gọi cho người thân hoặc bêu tên |
| obfuscated_filler | tin quảng cáo cờ bạc hoặc nhận thưởng bị chèn ký tự ngẫu nhiên, viết dính liền, để lọt qua bộ lọc, kèm một đường link |
| traffic_fine | báo phương tiện vi phạm giao thông (phạt nguội) chưa nộp phạt, yêu cầu tra cứu hoặc nộp qua link |
| job_task | tuyển cộng tác viên làm việc tại nhà, xử lý đơn hàng hoặc nhiệm vụ online, hứa thu nhập cao mỗi ngày |
| reward_points_expiry | báo điểm thưởng ngân hàng hoặc ví điện tử sắp hết hạn, yêu cầu đổi điểm qua link |
| bank_account_verification | giả ngân hàng báo tài khoản bị khoá, cần xác minh hoặc cập nhật thông tin qua link |
| loan_offer | mời vay tiền nhanh, giải ngân trong ngày, chỉ cần căn cước |
| parcel_delivery | giả đơn vị giao hàng báo không giao được bưu kiện, yêu cầu xác nhận địa chỉ hoặc trả phí qua link |
| identity_vneid | giả cơ quan nhà nước yêu cầu cập nhật định danh điện tử, VNeID hoặc căn cước qua link |
| sim_subscriber | giả nhà mạng báo thuê bao sắp bị khoá vì chưa chuẩn hoá thông tin, yêu cầu làm qua link |
| tax_einvoice | giả cơ quan thuế báo hoàn thuế hoặc hoá đơn điện tử, yêu cầu khai thông tin qua link |
| social_insurance | giả bảo hiểm xã hội báo nhận trợ cấp hoặc cập nhật sổ, yêu cầu khai thông tin qua link |
| prize_lucky_draw | báo trúng thưởng, quà tri ân khách hàng, yêu cầu nộp phí hoặc khai thông tin để nhận |
| investment | mời đầu tư, tiền ảo hoặc chứng khoán, hứa lợi nhuận cao và nhanh |
| impersonated_relative | giả người thân hoặc người quen nhờ chuyển tiền gấp hoặc báo người thân gặp nạn |

### Ham

| scenario | scenario_text |
|---|---|
| carrier_promo | nhà mạng giới thiệu gói cước, khuyến mãi nạp thẻ hoặc dịch vụ giá trị gia tăng, hướng dẫn soạn tin đăng ký |
| otp_code | mã OTP hoặc mã xác minh của một dịch vụ, dặn không chia sẻ cho ai |
| carrier_account_status | nhà mạng báo gói cước gia hạn, đăng ký thành công hoặc thất bại, tài khoản sắp hết, điểm được cộng |
| public_service_notice | thông báo của cơ quan nhà nước hoặc chính quyền địa phương: tuyên truyền, bầu cử, an toàn giao thông, hồ sơ dịch vụ công, thời tiết |
| subscriber_verification | nhà mạng thật nhắc xác thực thông tin thuê bao theo quy định, hướng dẫn làm trên ứng dụng chính thức hoặc tại cửa hàng |
| registered_advertising_qc | quảng cáo hợp pháp có tiền tố QC ở đầu tin, ví dụ truyền hình, internet, khuyến mãi |
| transaction_alert | ngân hàng báo biến động số dư hoặc giao dịch thẻ vừa thực hiện |
| personal | tin nhắn cá nhân giữa người quen, ngắn, đời thường |
| bill_reminder | nhắc thanh toán hoá đơn internet, điện, nước hoặc ví trả sau đến hạn |
| appointment_health | bệnh viện hoặc phòng khám báo lịch khám, lịch tái khám hoặc kết quả xét nghiệm |
| account_security_notice | dịch vụ báo đăng nhập sai nhiều lần, kích hoạt bảo mật hoặc thay đổi thiết lập tài khoản |
| delivery_status | đơn vị giao hàng thật báo đơn đã đến bưu cục hoặc đang giao |
| order_confirmation | sàn thương mại điện tử hoặc cửa hàng xác nhận đơn hàng đã đặt |

## Style fields

Drawn independently of class, so that none of them carries the label.

| field | values, equal probability |
|---|---|
| sender_text | `tên thương hiệu hoặc cơ quan (brandname)`, `đầu số ngắn`, `số điện thoại cá nhân` |
| length_text | `ngắn (dưới 80 ký tự)`, `vừa (80 đến 200 ký tự)`, `dài (200 đến 400 ký tự)` |
| accent_text | `tiếng Việt có dấu`, `tiếng Việt không dấu` |
