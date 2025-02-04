/**
 * The contents of this file are subject to the Mozilla Public License Version 1.1 (the "License");
 * you may not use this file except in compliance with the License. You may obtain a copy of the
 * License at http://www.mozilla.org/MPL/
 *
 * <p>Software distributed under the License is distributed on an "AS IS" basis, WITHOUT WARRANTY OF
 * ANY KIND, either express or implied. See the License for the specific language governing rights
 * and limitations under the License.
 *
 * <p>The Original Code is OpenELIS code.
 *
 * <p>Copyright (C) CIRG, University of Washington, Seattle WA. All Rights Reserved.
 */
package org.openelisglobal.reports.action.implementation;

import org.openelisglobal.internationalization.MessageUtil;

public class IndicatorAllTestLNSP extends IndicatorAllTest implements IReportCreator, IReportParameterSetter {

    @Override
    protected String getLabNameLine1() {
        return MessageUtil.getContextualMessage("report.labName.one");
    }

    @Override
    protected String getLabNameLine2() {
        return MessageUtil.getContextualMessage("report.labName.two");
    }
}
