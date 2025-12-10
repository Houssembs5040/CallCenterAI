"""Generate traffic for testing monitoring"""
import requests
import time
import random

API_URL = "http://localhost:6000/predict"

test_tickets = [
    "My laptop screen is broken and needs repair",
    "I forgot my password and cannot login",
    "Need to purchase new software licenses",
    "The database storage is full",
    "My email account is locked",
    "Computer won't turn on",
    "Need administrative access to install software",
    "Request for new hardware equipment",

]
test_tickets.extend([

    "After the most recent OS update, my workstation started exhibiting a strange intermittent issue where applications freeze for several seconds before recovering, and the system logs show repeated warnings about kernel I/O scheduling conflicts; I need assistance diagnosing whether this is hardware degradation, corrupted drivers, or a misconfiguration in the new update.",

    "Desde hace varios días, mi ordenador portátil se reinicia de forma inesperada cada vez que intento abrir aplicaciones pesadas como AutoCAD o MATLAB, y aunque actualicé los controladores de la tarjeta gráfica y ejecuté un análisis completo del sistema, el problema persiste; solicito una revisión exhaustiva del hardware y recomendaciones para evitar pérdidas de información.",
    

    "Mon compte de messagerie professionnel a été verrouillé après plusieurs tentatives de connexion échouées, probablement dues à un problème de synchronisation avec l’authentification multifactorielle; j’ai besoin d’une réinitialisation d’urgence ainsi que d’une vérification complète de la configuration de sécurité afin d’éviter que cela ne se reproduise.",
    
    "Seit der letzten Netzwerkumstellung erhalte ich ständig Fehlermeldungen beim Zugriff auf interne Ressourcen, insbesondere bei Datenbanken, und mein VPN trennt sich sporadisch ohne erkennbaren Grund; ich benötige eine vollständige Analyse der Netzwerkkonfiguration sowie Unterstützung bei der Wiederherstellung einer stabilen Verbindung.",
    

    "最新のソフトウェアアップデートを適用した後、業務用アプリケーションが異常終了を繰り返すようになり、ログにはメモリ関連のエラーやデータアクセス例外が頻出しています。業務に支障が出ているため、原因調査と早急な修正をお願いします。",
    

    "최근 데이터베이스 서버에 접속할 때마다 응답 지연이 심해지고 간헐적으로 연결이 끊기는 현상이 발생합니다. 네트워크 패킷 손실이 의심되며, 서버 리소스 모니터링 결과 CPU 스파이크도 반복적으로 나타나고 있습니다. 종합적인 점검 및 성능 최적화가 필요합니다.",
    

    "मेरे कार्यालय के कंप्यूटर पर बैकअप सॉफ़्टवेयर ठीक से काम नहीं कर रहा है और हर रात निर्धारित बैकअप असफल हो जाता है। लॉग फ़ाइलों में डिस्क एक्सेस त्रुटियाँ और अनुमति से जुड़े मुद्दे दिखाई दे रहे हैं। कृपया इस समस्या की विस्तृत जाँच करें और सुनिश्चित करें कि आगे से बैकअप सफलतापूर्वक निष्पादित हो।",
    

    "Após configurar um novo conjunto de políticas de acesso, vários usuários relataram que perderam permissões para sistemas essenciais, e algumas integrações de API pararam de funcionar completamente. Preciso de uma revisão detalhada das permissões, logs de autenticação e das configurações de IAM.",

    "بعد تثبيت برنامج الحماية الجديد على جهاز الحاسوب الخاص بي، بدأت أواجه بطئاً شديداً في تشغيل التطبيقات بالإضافة إلى رسائل خطأ متكررة تتعلق بتعارضات بين البرامج الأمنية. أحتاج إلى تحليل شامل للمشكلة وإعادة ضبط الإعدادات بما يضمن أداءً مستقراً.",
    

    "My workstation suddenly loses connectivity to internal services, and although the Wi-Fi indicator shows full signal strength, I get intermittent DNS failures; además, desde la actualización del firmware, el dispositivo empieza a calentarse mucho más de lo normal, что делает работу практически невозможной, so I urgently need a detailed diagnosis across networking, hardware, and power-management subsystems."
])


def send_request():
    """Send a single request"""
    ticket = random.choice(test_tickets)
    try:
        response = requests.post(
            API_URL,
            json={"text": ticket, "return_probas": True},
            timeout=10
        )
        print(f"✅ {response.status_code} - {ticket[:50]}...")
    except Exception as e:
        print(f"❌ Error: {e}")

if __name__ == "__main__":
    print("🚀 Generating traffic...")
    print("Press Ctrl+C to stop\n")
    
    try:
        while True:
            send_request()
            time.sleep(random.uniform(0.5, 2.0))
    except KeyboardInterrupt:
        print("\n✅ Stopped!")