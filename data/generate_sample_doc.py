"""
Generates a synthetic but realistic multi-page 'employee handbook' PDF
for Northwind Robotics (a fictional company). This gives DocMind a real
document to ingest, chunk, and answer questions about, with specific
facts/numbers we can write unambiguous gold Q&A pairs against later.

Run: python generate_sample_doc.py
"""

from fpdf import FPDF

SECTIONS = [
    ("1. Company Overview", """
Northwind Robotics was founded in 2016 and is headquartered in Austin, Texas,
with a satellite engineering office in Pune, India. The company designs and
manufactures autonomous warehouse robots used by mid-size logistics providers.
As of the most recent fiscal year, Northwind Robotics employs 340 people
across engineering, sales, operations, and customer support. The company's
mission statement is "to make warehouse automation affordable for businesses
that were previously priced out of robotics." Northwind Robotics is a private
company and has not filed for an IPO as of this handbook's publication date.
"""),
    ("2. Work Hours and Remote Work Policy", """
Standard working hours are 9:00 AM to 6:00 PM local time, Monday through
Friday, with a one-hour unpaid lunch break. Employees are expected to be
online and reachable during core hours of 11:00 AM to 4:00 PM local time.
Northwind Robotics operates a hybrid work model: employees are required to
work from the office a minimum of 2 days per week, with Tuesday designated
as a mandatory in-office day for all teams. Fully remote work arrangements
must be approved in writing by both the employee's manager and the VP of
People Operations, and are re-evaluated every 6 months. Employees working
from outside their home country for more than 30 consecutive days must
notify the People Operations team in advance for tax and compliance reasons.
"""),
    ("3. Paid Time Off (PTO) and Leave Policy", """
Full-time employees accrue 18 days of paid time off (PTO) per calendar year,
accrued at a rate of 1.5 days per completed month of employment. PTO can be
carried over into the following year up to a maximum of 5 unused days;
anything beyond that is forfeited on December 31st unless local law requires
otherwise. New employees become eligible to use accrued PTO after completing
a 90-day probationary period. Northwind Robotics also provides 10 paid
holidays per year, published annually by People Operations. Parental leave
is 16 weeks fully paid for the primary caregiver and 6 weeks fully paid for
the secondary caregiver, regardless of gender, and can be taken within 12
months of the birth or adoption. Employees requesting PTO of 5 or more
consecutive days must submit the request at least 14 days in advance for
manager approval.
"""),
    ("4. Health and Wellness Benefits", """
Northwind Robotics covers 100% of the employee's health insurance premium
and 70% of the premium for dependents under the company's PPO health plan.
Employees become eligible for health benefits on their first day of
employment; there is no waiting period. The company provides a monthly
wellness stipend of $75 that can be used for gym memberships, fitness
classes, or mental health app subscriptions, reimbursed through the expense
system. Northwind Robotics also offers an Employee Assistance Program (EAP)
providing up to 6 free confidential counseling sessions per employee per
year, available 24/7 through a third-party provider.
"""),
    ("5. Expense Reimbursement Policy", """
Business-related expenses must be submitted through the Expensify system
within 30 days of the expense being incurred; expenses submitted after 60
days will not be reimbursed except in extraordinary circumstances approved
by Finance. Any single expense over $150 requires a manager's pre-approval
before the purchase is made. Meal expenses while traveling on company
business are capped at $60 per day domestically and $90 per day
internationally. Airfare must be booked in economy class for flights under
6 hours; business class is permitted for flights exceeding 6 hours with
director-level approval or higher. Reimbursements are processed on a
bi-weekly basis, on the same schedule as payroll.
"""),
    ("6. Equipment and IT Policy", """
All new employees are issued a company laptop within their first 3 business
days. Engineering employees receive a laptop with a minimum of 32GB RAM;
all other roles receive a standard 16GB configuration. Employees may request
an equipment refresh every 3 years, or sooner if a device fails and cannot
be repaired. Lost or stolen company equipment must be reported to the IT
Security team within 24 hours of discovery. Installing unauthorized
third-party software on company laptops is prohibited without prior approval
from IT Security. All company laptops are enrolled in mobile device
management (MDM) and are remotely wipeable in the event of loss, theft, or
employee offboarding.
"""),
    ("7. Code of Conduct and Anti-Harassment Policy", """
Northwind Robotics maintains a zero-tolerance policy toward harassment,
discrimination, and retaliation of any kind. Employees who witness or
experience a violation of this policy are encouraged to report it to their
manager, any member of People Operations, or through the anonymous ethics
hotline available at ethics.northwindrobotics.example. All reports are
investigated within 10 business days of being received. Retaliation against
an employee for making a good-faith report is itself grounds for
termination. Confidentiality is maintained to the extent possible during
an investigation, consistent with the need to conduct a thorough and fair
review.
"""),
    ("8. Performance Review Process", """
Formal performance reviews are conducted twice a year, in June and
December. Each review cycle includes a self-assessment, peer feedback from
2-4 colleagues selected by the employee and manager jointly, and a manager
evaluation. Employees are rated on a 5-point scale ranging from "Does Not
Meet Expectations" to "Significantly Exceeds Expectations." Compensation
adjustments and promotions are considered during the December review cycle
only, based on the prior 12 months of performance. Employees who receive a
rating of "Does Not Meet Expectations" are placed on a Performance
Improvement Plan (PIP) lasting 60 days, with a follow-up review at the end
of that period.
"""),
    ("9. Data Security and Confidentiality Policy", """
All employees must complete mandatory security awareness training within
their first 2 weeks of employment and annually thereafter. Access to
customer data is granted on a least-privilege, role-based basis and is
reviewed quarterly by the Security team. Employees must use multi-factor
authentication (MFA) for all company systems, including email, source
control, and cloud infrastructure. Any suspected data breach or security
incident must be reported to security@northwindrobotics.example within 1
hour of discovery. Confidential company information, including source code,
customer lists, and unreleased product designs, may not be shared outside
the company without written authorization from the employee's Vice
President.
"""),
    ("10. Resignation and Termination Policy", """
Employees voluntarily resigning are asked to provide a minimum of 2 weeks'
written notice; engineering managers and above are asked to provide 4
weeks' notice to allow for knowledge transfer. Final paychecks, including
any unused accrued PTO payout, are issued within 10 business days of the
last working day, or sooner where required by local law. Company equipment
must be returned on or before the last working day. Employees terminated
for cause forfeit any unused PTO payout in states where this is legally
permitted. Exit interviews are offered, but not mandatory, for all
departing employees and are conducted by a People Operations representative
who is not the employee's direct manager.
"""),
]


def build_pdf(path: str):
    pdf = FPDF(format="A4")
    pdf.set_margins(15, 15, 15)
    pdf.set_auto_page_break(auto=True, margin=20)
    pdf.add_page()
    w = pdf.w - pdf.l_margin - pdf.r_margin

    pdf.set_font("Helvetica", "B", 18)
    pdf.multi_cell(w, 10, "Northwind Robotics", align="C")
    pdf.set_font("Helvetica", "", 13)
    pdf.multi_cell(w, 8, "Employee Handbook", align="C")
    pdf.ln(8)

    for heading, body in SECTIONS:
        pdf.set_font("Helvetica", "B", 13)
        pdf.multi_cell(w, 8, heading)
        pdf.ln(1)
        pdf.set_font("Helvetica", "", 11)
        pdf.multi_cell(w, 6, body.strip())
        pdf.ln(4)

    pdf.output(path)
    print(f"Wrote {path}")


if __name__ == "__main__":
    build_pdf("employee_handbook.pdf")
