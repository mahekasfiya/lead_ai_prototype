from module_3.intelligence.contact_extractor import ContactExtractor


def test_extracts_email_phone_name_and_department():
    extractor = ContactExtractor()

    text = """
    Contact Person: Sarah Ahmed
    Procurement Department
    Email: sarah.ahmed@abcbank.ae
    Telephone: +971 6 555 1234
    Tender portal:
    https://procurement.abcbank.ae/tenders/tafj-migration
    """

    contacts = extractor.extract(
        text=text,
        source_url="https://abcbank.ae",
    )

    assert len(contacts) >= 1

    contact = contacts[0]

    assert contact.name == "Sarah Ahmed"
    assert contact.email == "sarah.ahmed@abcbank.ae"
    assert contact.department == "Procurement"

def test_returns_empty_list_when_no_contacts_exist():
    extractor = ContactExtractor()

    contacts = extractor.extract(
        text="The organization plans to modernize its ERP environment.",
        source_url="https://company.com/news",
    )

    assert contacts == []