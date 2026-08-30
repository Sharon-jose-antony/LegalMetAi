"""
LEGALMET AI — End-to-End API and Integration Verification Script
"""
import urllib.request
import json
import os

base = os.getenv("API_URL", "http://127.0.0.1:8000")

def test_all():
    print("=" * 60)
    print("LEGALMET AI — End-to-End Verification Suite")
    print("=" * 60)

    # 1. Test root HTML
    req = urllib.request.Request(f"{base}/")
    with urllib.request.urlopen(req) as res:
        html = res.read().decode('utf-8')
        assert 'LEGALMET AI' in html
        print(f"[PASS] 1. Root HTML served correctly ({len(html)} bytes)")

    # 2. Test login
    login_data = json.dumps({'username': 'admin', 'password': 'Admin@1234!'}).encode('utf-8')
    req = urllib.request.Request(f"{base}/api/auth/login", data=login_data, headers={'Content-Type': 'application/json'})
    with urllib.request.urlopen(req) as res:
        auth = json.loads(res.read().decode('utf-8'))
        token = auth['access_token']
        print(f"[PASS] 2. Login successful (admin token received)")

    # 3. Test Dashboard Stats
    req = urllib.request.Request(f"{base}/api/dashboard/stats", headers={'Authorization': f'Bearer {token}'})
    with urllib.request.urlopen(req) as res:
        stats = json.loads(res.read().decode('utf-8'))
        total = stats['total_inspections']
        passes = stats['pass_count']
        nc = stats['potential_non_compliance_count']
        mr = stats['manual_review_count']
        print(f"[PASS] 3. Dashboard stats verified: Total={total}, Pass={passes}, Non-Compliance={nc}, Manual Review={mr}")

    # 4. Test Inspections List
    req = urllib.request.Request(f"{base}/api/inspections", headers={'Authorization': f'Bearer {token}'})
    with urllib.request.urlopen(req) as res:
        inspections = json.loads(res.read().decode('utf-8'))
        print(f"[PASS] 4. Inspections list loaded ({len(inspections)} items)")
        for i in inspections[:3]:
            print(f"       • {i['inspection_id']}: {i['product_name']} | Status: {i['status']} | Conf: {i['overall_confidence']}")

    # 5. Test Demo Detail (LM-2026-DEMO01)
    req = urllib.request.Request(f"{base}/api/inspections/LM-2026-DEMO01", headers={'Authorization': f'Bearer {token}'})
    with urllib.request.urlopen(req) as res:
        demo1 = json.loads(res.read().decode('utf-8'))
        assert demo1['status'] == 'PASS'
        print(f"[PASS] 5. Demo 1 (Compliant Food) verified: Status={demo1['status']}, Conf={demo1['overall_confidence']:.2f}, Decls={len(demo1['declarations'])}")

    # 6. Test Demo Detail (LM-2026-DEMO02)
    req = urllib.request.Request(f"{base}/api/inspections/LM-2026-DEMO02", headers={'Authorization': f'Bearer {token}'})
    with urllib.request.urlopen(req) as res:
        demo2 = json.loads(res.read().decode('utf-8'))
        assert demo2['status'] == 'MANUAL_REVIEW'
        print(f"[PASS] 6. Demo 2 (Glare/Cosmetics) verified: Status={demo2['status']}, Conf={demo2['overall_confidence']:.2f}")

    # 7. Test Demo Detail (LM-2026-DEMO03)
    req = urllib.request.Request(f"{base}/api/inspections/LM-2026-DEMO03", headers={'Authorization': f'Bearer {token}'})
    with urllib.request.urlopen(req) as res:
        demo3 = json.loads(res.read().decode('utf-8'))
        assert demo3['status'] == 'POTENTIAL_NON_COMPLIANCE'
        print(f"[PASS] 7. Demo 3 (Non-compliant Household) verified: Status={demo3['status']}, Conf={demo3['overall_confidence']:.2f}")

    # 8. Test Manual Review Submission
    review_data = json.dumps({
        'decision': 'CONFIRMED',
        'statutory_notes': 'Verification notes recorded during automated testing suite.',
        'flagged_declarations': ['CONSUMER_CARE_DETAILS', 'UNIT_SALE_PRICE']
    }).encode('utf-8')
    req = urllib.request.Request(f"{base}/api/inspections/LM-2026-DEMO03/review", data=review_data, headers={
        'Authorization': f'Bearer {token}',
        'Content-Type': 'application/json'
    })
    with urllib.request.urlopen(req) as res:
        rev_res = json.loads(res.read().decode('utf-8'))
        assert rev_res['decision'] == 'CONFIRMED'
        print(f"[PASS] 8. Inspector Manual Review submitted and saved (Decision: {rev_res['decision']})")

    # 9. Test PDF Download
    req = urllib.request.Request(f"{base}/api/inspections/LM-2026-DEMO01/report.pdf", headers={'Authorization': f'Bearer {token}'})
    with urllib.request.urlopen(req) as res:
        pdf_bytes = res.read()
        assert pdf_bytes.startswith(b'%PDF')
        print(f"[PASS] 9. Inspection PDF report generated & downloaded ({len(pdf_bytes)} bytes)")

    print("\n" + "=" * 60)
    print("ALL 9 INTEGRATION & ENDPOINT CHECKS PASSED (100% SUCCESS)!")
    print("=" * 60)

if __name__ == '__main__':
    test_all()
